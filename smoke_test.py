from kfp import dsl, Client

USER = "user@example.com"
NAMESPACE = "yolo-project"

@dsl.component(base_image="python:3.11-slim")
def say_hello(name: str) -> str:
    print(f"Hello, {name}")
    return name

@dsl.pipeline(name="smoke-test")
def smoke_test(name: str = "kubeflow"):
    say_hello(name=name)

client = Client(host="http://localhost:8888", namespace=NAMESPACE)

# Multi-user mode needs the identity header that Istio normally adds.
# Set it on every generated API client the SDK created.
for attr in dir(client):
    api = getattr(client, attr, None)
    session = getattr(api, "api_client", None)
    if session is not None and hasattr(session, "default_headers"):
        session.default_headers["kubeflow-userid"] = USER

run = client.create_run_from_pipeline_func(
    smoke_test,
    arguments={"name": "yolo"},
    experiment_name="smoke",
)
print("run id:", run.run_id)
