from kfp import dsl, compiler
from kfp.dsl import Output, Input, Dataset


@dsl.component(base_image="python:3.11-slim")
def produce(text: str, out: Output[Dataset]):
    # Writes a file. KFP uploads it to minio after this step finishes.
    with open(out.path, "w") as f:
        f.write(text)


@dsl.component(base_image="python:3.11-slim")
def consume(inp: Input[Dataset]):
    # Downloads the file from minio before this step starts.
    with open(inp.path) as f:
        print("READ FROM ARTIFACT:", f.read())


@dsl.pipeline(name="artifact-test")
def artifact_test(text: str = "hello artifacts"):
    p = produce(text=text)
    consume(inp=p.outputs["out"])


if __name__ == "__main__":
    compiler.Compiler().compile(artifact_test, "artifact_test.yaml")
    print("wrote artifact_test.yaml")