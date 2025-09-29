"""
Zendesk MCP Tools

MCP tools for Zendesk integration including queue polling, ticket retrieval,
and status updates. These tools are used by the Poller Agent to monitor
Zendesk queues and process tickets.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import aiohttp
from dataclasses import dataclass

from .base_server import BaseMCPServer


@dataclass
class ZendeskConfig:
    """Zendesk configuration for a team.

    Supports either API token auth (email + api_token) or basic auth (username + password).
    Subdomain is optional; if not provided, will attempt to read from the
    ZENDESK_SUBDOMAIN env var or use a provided base_url.
    """
    team_id: str
    # Auth options
    email: Optional[str] = None
    api_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    # Endpoint options
    subdomain: Optional[str] = None
    base_url: Optional[str] = None
    # Defaults
    poll_interval: int = 30  # seconds
    max_tickets_per_poll: int = 50
    ticket_fields: List[str] = None

    def __post_init__(self):
        if self.ticket_fields is None:
            self.ticket_fields = [
                "id", "subject", "status", "priority", "assignee_id",
                "requester_id", "created_at", "updated_at", "tags", "description"
            ]


class ZendeskClient:
    """Zendesk API client for queue polling and ticket operations."""
    
    def __init__(self, config: ZendeskConfig):
        self.config = config
        self.base_url = self._compute_base_url(config)
        self.auth = self._compute_auth(config)
        self.session = None

    def _compute_base_url(self, config: ZendeskConfig) -> str:
        import os
        if config.base_url:
            return config.base_url.rstrip("/")
        subdomain = config.subdomain or os.getenv("ZENDESK_SUBDOMAIN")
        if not subdomain:
            raise ValueError("Zendesk subdomain or base_url is required. Set ZENDESK_SUBDOMAIN or provide base_url.")
        return f"https://{subdomain}.zendesk.com/api/v2"

    def _compute_auth(self, config: ZendeskConfig):
        # Prefer token auth if provided
        if config.email and config.api_token:
            return (f"{config.email}/token", config.api_token)
        # Fallback to basic auth
        if config.username and config.password:
            return (config.username, config.password)
        raise ValueError("Zendesk credentials required: either email+api_token or username+password")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            auth=aiohttp.BasicAuth(*self.auth),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_queue_tickets(
        self, 
        queue_name: str, 
        status: str = "new",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get tickets from a specific Zendesk queue.
        
        Args:
            queue_name: Name of the queue to poll
            status: Ticket status to filter by (new, open, pending, solved, closed)
            limit: Maximum number of tickets to retrieve
            
        Returns:
            List of ticket dictionaries
        """
        try:
            # Build search query for the queue
            query = f"type:ticket status:{status}"
            if queue_name != "all":
                query += f" queue:{queue_name}"
            
            url = f"{self.base_url}/search.json"
            params = {
                "query": query,
                "sort_by": "created_at",
                "sort_order": "asc",
                "per_page": limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("results", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Zendesk API error: {response.status} - {error_text}")
        
        except Exception as e:
            print(f"Error getting queue tickets: {e}")
            return []
    
    async def get_ticket_details(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific ticket.
        
        Args:
            ticket_id: Zendesk ticket ID
            
        Returns:
            Ticket details dictionary or None if not found
        """
        try:
            url = f"{self.base_url}/tickets/{ticket_id}.json"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("ticket", {})
                else:
                    return None
        
        except Exception as e:
            print(f"Error getting ticket details: {e}")
            return None
    
    async def update_ticket_status(
        self, 
        ticket_id: str, 
        status: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        Update ticket status and optionally add a comment.
        
        Args:
            ticket_id: Zendesk ticket ID
            status: New status (new, open, pending, solved, closed)
            comment: Optional comment to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/tickets/{ticket_id}.json"
            
            update_data = {
                "ticket": {
                    "status": status
                }
            }
            
            if comment:
                update_data["ticket"]["comment"] = {
                    "body": comment,
                    "public": True
                }
            
            async with self.session.put(url, json=update_data) as response:
                return response.status == 200
        
        except Exception as e:
            print(f"Error updating ticket status: {e}")
            return False
    
    async def assign_ticket(
        self, 
        ticket_id: str, 
        assignee_id: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        Assign a ticket to a specific user.
        
        Args:
            ticket_id: Zendesk ticket ID
            assignee_id: User ID to assign to
            comment: Optional assignment comment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/tickets/{ticket_id}.json"
            
            update_data = {
                "ticket": {
                    "assignee_id": assignee_id
                }
            }
            
            if comment:
                update_data["ticket"]["comment"] = {
                    "body": comment,
                    "public": True
                }
            
            async with self.session.put(url, json=update_data) as response:
                return response.status == 200
        
        except Exception as e:
            print(f"Error assigning ticket: {e}")
            return False
    
    async def add_ticket_tags(
        self, 
        ticket_id: str, 
        tags: List[str]
    ) -> bool:
        """
        Add tags to a ticket.
        
        Args:
            ticket_id: Zendesk ticket ID
            tags: List of tags to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/tickets/{ticket_id}.json"
            
            update_data = {
                "ticket": {
                    "tags": tags
                }
            }
            
            async with self.session.put(url, json=update_data) as response:
                return response.status == 200
        
        except Exception as e:
            print(f"Error adding ticket tags: {e}")
            return False
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Dictionary with queue statistics
        """
        try:
            # Get tickets by status
            stats = {
                "queue_name": queue_name,
                "timestamp": datetime.now().isoformat(),
                "counts": {}
            }
            
            statuses = ["new", "open", "pending", "solved", "closed"]
            
            for status in statuses:
                tickets = await self.get_queue_tickets(queue_name, status, limit=1)
                stats["counts"][status] = len(tickets)
            
            return stats
        
        except Exception as e:
            print(f"Error getting queue stats: {e}")
            return {"queue_name": queue_name, "error": str(e)}


class ZendeskTools:
    """MCP tools for Zendesk operations."""
    
    def __init__(self):
        self.configs: Dict[str, ZendeskConfig] = {}
        self.clients: Dict[str, ZendeskClient] = {}
    
    def register_team_config(self, team: str, config: ZendeskConfig):
        """Register Zendesk configuration for a team (one client per team)."""
        self.configs[team] = config
        self.clients[team] = ZendeskClient(config)
    
    async def poll_queue(
        self,
        team: str,
        queue_name: Optional[str] = None,
        status: str = "new",
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Poll a Zendesk queue for new tickets.
        
        Args:
            team: Team name
            queue_name: Queue name (uses team default if not provided)
            status: Ticket status to poll for
            limit: Maximum tickets to retrieve
            
        Returns:
            Dictionary with poll results
        """
        try:
            if team not in self.configs:
                return {
                    "success": False,
                    "error": f"No Zendesk configuration found for team: {team}"
                }
            
            config = self.configs[team]
            client = self.clients[team]
            
            # Use provided queue_name or team default
            target_queue = queue_name or config.queue_name
            
            async with client as zendesk:
                tickets = await zendesk.get_queue_tickets(
                    queue_name=target_queue,
                    status=status,
                    limit=limit
                )
                
                return {
                    "success": True,
                    "team": team,
                    "queue": target_queue,
                    "status": status,
                    "ticket_count": len(tickets),
                    "tickets": tickets,
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "team": team,
                "queue": queue_name
            }
    
    async def get_ticket_details(
        self,
        team: str,
        ticket_id: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific ticket.
        
        Args:
            team: Team name
            ticket_id: Zendesk ticket ID
            
        Returns:
            Dictionary with ticket details
        """
        try:
            if team not in self.configs:
                return {
                    "success": False,
                    "error": f"No Zendesk configuration found for team: {team}"
                }
            
            client = self.clients[team]
            
            async with client as zendesk:
                ticket = await zendesk.get_ticket_details(ticket_id)
                
                if ticket:
                    return {
                        "success": True,
                        "team": team,
                        "ticket_id": ticket_id,
                        "ticket": ticket,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": "Ticket not found",
                        "team": team,
                        "ticket_id": ticket_id
                    }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "team": team,
                "ticket_id": ticket_id
            }
    
    async def update_ticket_status(
        self,
        team: str,
        ticket_id: str,
        status: str,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update ticket status.
        
        Args:
            team: Team name
            ticket_id: Zendesk ticket ID
            status: New status
            comment: Optional comment
            
        Returns:
            Dictionary with update result
        """
        try:
            if team not in self.configs:
                return {
                    "success": False,
                    "error": f"No Zendesk configuration found for team: {team}"
                }
            
            client = self.clients[team]
            
            async with client as zendesk:
                success = await zendesk.update_ticket_status(
                    ticket_id=ticket_id,
                    status=status,
                    comment=comment
                )
                
                return {
                    "success": success,
                    "team": team,
                    "ticket_id": ticket_id,
                    "status": status,
                    "comment": comment,
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "team": team,
                "ticket_id": ticket_id
            }
    
    async def assign_ticket(
        self,
        team: str,
        ticket_id: str,
        assignee_id: str,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assign ticket to a user.
        
        Args:
            team: Team name
            ticket_id: Zendesk ticket ID
            assignee_id: User ID to assign to
            comment: Optional assignment comment
            
        Returns:
            Dictionary with assignment result
        """
        try:
            if team not in self.configs:
                return {
                    "success": False,
                    "error": f"No Zendesk configuration found for team: {team}"
                }
            
            client = self.clients[team]
            
            async with client as zendesk:
                success = await zendesk.assign_ticket(
                    ticket_id=ticket_id,
                    assignee_id=assignee_id,
                    comment=comment
                )
                
                return {
                    "success": success,
                    "team": team,
                    "ticket_id": ticket_id,
                    "assignee_id": assignee_id,
                    "comment": comment,
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "team": team,
                "ticket_id": ticket_id
            }
    
    async def get_queue_stats(
        self,
        team: str,
        queue_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Args:
            team: Team name
            queue_name: Queue name (uses team default if not provided)
            
        Returns:
            Dictionary with queue statistics
        """
        try:
            if team not in self.configs:
                return {
                    "success": False,
                    "error": f"No Zendesk configuration found for team: {team}"
                }
            
            config = self.configs[team]
            client = self.clients[team]
            
            # Use provided queue_name or team default
            target_queue = queue_name or config.queue_name
            
            async with client as zendesk:
                stats = await zendesk.get_queue_stats(target_queue)
                
                return {
                    "success": True,
                    "team": team,
                    "queue": target_queue,
                    "stats": stats,
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "team": team,
                "queue": queue_name
            }


class ZendeskMCPServer(BaseMCPServer):
    """
    MCP Server for Zendesk operations.
    
    Exposes Zendesk tools through the MCP protocol for use by
    the Poller Agent and other agents.
    """
    
    def __init__(self, port: int = 8088):
        super().__init__(port=port)
        self.zendesk_tools = ZendeskTools()
        
        # Register Zendesk tools
        self.register_tool("poll_queue", self.zendesk_tools.poll_queue)
        self.register_tool("get_ticket_details", self.zendesk_tools.get_ticket_details)
        self.register_tool("update_ticket_status", self.zendesk_tools.update_ticket_status)
        self.register_tool("assign_ticket", self.zendesk_tools.assign_ticket)
        self.register_tool("get_queue_stats", self.zendesk_tools.get_queue_stats)
        self.register_tool("add_ticket_tags", self.zendesk_tools.add_ticket_tags)
    
    def register_team_config(self, team: str, config: ZendeskConfig):
        """Register Zendesk configuration for a team."""
        self.zendesk_tools.register_team_config(team, config)
    
    async def start(self):
        """Start the Zendesk MCP server."""
        await super().start()
        print(f"🎫 Zendesk MCP Server started on port {self.port}")
        print("Available tools:")
        print("  - poll_queue: Poll Zendesk queue for new tickets")
        print("  - get_ticket_details: Get detailed ticket information")
        print("  - update_ticket_status: Update ticket status")
        print("  - assign_ticket: Assign ticket to user")
        print("  - get_queue_stats: Get queue statistics")
