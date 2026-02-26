from livekit import api

token = (
    api.AccessToken("devkey", "secret")
    .with_identity("viewer")
    .with_grants(api.VideoGrants(room_join=True, room="robot-stream"))
    .to_jwt()
)
print(token)
