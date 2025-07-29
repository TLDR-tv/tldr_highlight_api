"""Data models for Twitch EventSub events.

This module contains Pydantic models for all supported Twitch EventSub event types
and their associated data structures.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class EventSubTransport(BaseModel):
    """EventSub transport configuration."""
    method: str
    session_id: Optional[str] = None
    callback: Optional[str] = None
    secret: Optional[str] = None
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None


class EventSubCondition(BaseModel):
    """EventSub subscription condition."""
    model_config = ConfigDict(extra="allow")  # Allow additional fields
    
    broadcaster_user_id: Optional[str] = None
    user_id: Optional[str] = None
    moderator_user_id: Optional[str] = None
    to_broadcaster_user_id: Optional[str] = None
    from_broadcaster_user_id: Optional[str] = None
    reward_id: Optional[str] = None


class EventSubSubscription(BaseModel):
    """EventSub subscription details."""
    id: str
    status: str
    type: str
    version: str
    condition: EventSubCondition
    transport: EventSubTransport
    created_at: datetime
    cost: int


# Message-related models

class Badge(BaseModel):
    """Chat badge information."""
    set_id: str
    id: str
    info: str


class Emote(BaseModel):
    """Chat emote information."""
    id: str
    emote_set_id: str
    owner_id: Optional[str] = None
    format: List[str] = Field(default_factory=list)


class MessageFragment(BaseModel):
    """Chat message fragment."""
    type: str
    text: str
    cheermote: Optional[Dict[str, Any]] = None
    emote: Optional[Dict[str, Any]] = None
    mention: Optional[Dict[str, Any]] = None


class CheerTier(BaseModel):
    """Cheer tier information."""
    min_bits: int
    id: str
    color: str
    images: Dict[str, Dict[str, str]]
    can_cheer: bool
    show_in_bits_card: bool


class Message(BaseModel):
    """Chat message details."""
    text: str
    fragments: Optional[List[MessageFragment]] = None


class Reply(BaseModel):
    """Reply parent message information."""
    parent_message_id: str
    parent_message_body: str
    parent_user_id: str
    parent_user_name: str
    parent_user_login: str
    thread_message_id: str
    thread_user_id: str
    thread_user_name: str
    thread_user_login: str


# Event data models

class ChannelChatMessageEvent(BaseModel):
    """channel.chat.message event data."""
    broadcaster_user_id: str
    broadcaster_user_login: str
    broadcaster_user_name: str
    chatter_user_id: str
    chatter_user_login: str
    chatter_user_name: str
    message_id: str
    message: Message
    color: str
    badges: List[Badge]
    message_type: str
    cheer: Optional[Dict[str, Any]] = None
    reply: Optional[Reply] = None
    channel_points_custom_reward_id: Optional[str] = None
    channel_points_animation_id: Optional[str] = None


class ChannelFollowEvent(BaseModel):
    """channel.follow event data."""
    user_id: str
    user_login: str
    user_name: str
    broadcaster_user_id: str
    broadcaster_user_login: str
    broadcaster_user_name: str
    followed_at: datetime


class ChannelSubscribeEvent(BaseModel):
    """channel.subscribe event data."""
    user_id: str
    user_login: str
    user_name: str
    broadcaster_user_id: str
    broadcaster_user_login: str
    broadcaster_user_name: str
    tier: str
    is_gift: bool


class ChannelSubscriptionMessageEvent(BaseModel):
    """channel.subscription.message event data."""
    user_id: str
    user_login: str
    user_name: str
    broadcaster_user_id: str
    broadcaster_user_login: str
    broadcaster_user_name: str
    tier: str
    message: Message
    cumulative_months: int
    streak_months: Optional[int] = None
    duration_months: int


class ChannelCheerEvent(BaseModel):
    """channel.cheer event data."""
    is_anonymous: bool
    user_id: Optional[str] = None
    user_login: Optional[str] = None
    user_name: Optional[str] = None
    broadcaster_user_id: str
    broadcaster_user_login: str
    broadcaster_user_name: str
    message: str
    bits: int


class ChannelRaidEvent(BaseModel):
    """channel.raid event data."""
    from_broadcaster_user_id: str
    from_broadcaster_user_login: str
    from_broadcaster_user_name: str
    to_broadcaster_user_id: str
    to_broadcaster_user_login: str
    to_broadcaster_user_name: str
    viewers: int


class HypeTrainContribution(BaseModel):
    """Hype train contribution details."""
    user_id: str
    user_login: str
    user_name: str
    type: str  # bits, subscription, other
    total: int


class ChannelHypeTrainBeginEvent(BaseModel):
    """channel.hype_train.begin event data."""
    id: str
    broadcaster_user_id: str
    broadcaster_user_login: str
    broadcaster_user_name: str
    total: int
    progress: int
    goal: int
    top_contributions: List[HypeTrainContribution]
    last_contribution: HypeTrainContribution
    level: int
    started_at: datetime
    expires_at: datetime


class ChannelHypeTrainProgressEvent(BaseModel):
    """channel.hype_train.progress event data."""
    id: str
    broadcaster_user_id: str
    broadcaster_user_login: str
    broadcaster_user_name: str
    total: int
    progress: int
    goal: int
    top_contributions: List[HypeTrainContribution]
    last_contribution: HypeTrainContribution
    level: int
    started_at: datetime
    expires_at: datetime


class ChannelHypeTrainEndEvent(BaseModel):
    """channel.hype_train.end event data."""
    id: str
    broadcaster_user_id: str
    broadcaster_user_login: str
    broadcaster_user_name: str
    level: int
    total: int
    top_contributions: List[HypeTrainContribution]
    started_at: datetime
    ended_at: datetime
    cooldown_ends_at: datetime


# WebSocket message models

class SessionPayload(BaseModel):
    """WebSocket session payload."""
    id: str
    status: str
    connected_at: datetime
    keepalive_timeout_seconds: int
    reconnect_url: Optional[str] = None


class WelcomePayload(BaseModel):
    """Welcome message payload."""
    session: SessionPayload


class NotificationPayload(BaseModel):
    """Notification message payload."""
    subscription: EventSubSubscription
    event: Dict[str, Any]


class ReconnectPayload(BaseModel):
    """Reconnect message payload."""
    session: SessionPayload


class RevocationPayload(BaseModel):
    """Revocation message payload."""
    subscription: EventSubSubscription


class MessageType(str, Enum):
    """WebSocket message types."""
    SESSION_WELCOME = "session_welcome"
    SESSION_KEEPALIVE = "session_keepalive"
    SESSION_RECONNECT = "session_reconnect"
    NOTIFICATION = "notification"
    REVOCATION = "revocation"


class MessageMetadata(BaseModel):
    """WebSocket message metadata."""
    message_id: str
    message_type: MessageType
    message_timestamp: datetime
    subscription_type: Optional[str] = None
    subscription_version: Optional[str] = None


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    metadata: MessageMetadata
    payload: Optional[Union[
        WelcomePayload,
        NotificationPayload,
        ReconnectPayload,
        RevocationPayload,
        Dict[str, Any]
    ]] = None