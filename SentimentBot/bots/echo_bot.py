# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount


class EchoBot(ActivityHandler):
    def __init__(self, text_analytics_client):
        self.client = text_analytics_client
        
    async def on_members_added_activity(
        self, members_added: [ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")

    async def on_message_activity(self, turn_context: TurnContext):
        user_text = turn_context.activity.text
        result = self.client.analyze_sentiment(documents=[user_text])[0]
        return await turn_context.send_activity(
        MessageFactory.text(f"Echo: {user_text} (Sentiment: {result.sentiment})")
    )
