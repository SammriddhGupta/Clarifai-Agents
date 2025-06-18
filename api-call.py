from clarifai.client import Model

model = Model(
    url="https://clarifai.com/openai/chat-completion/models/o4-mini",
    # deployment_id="DEPLOYMENT_ID_HERE"
)

response = model.predict("What is a usecase for AI Agents for students?")
print(response)