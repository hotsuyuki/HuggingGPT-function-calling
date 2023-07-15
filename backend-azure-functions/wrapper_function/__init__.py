import os
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Annotated

from .hugging_gpt import hugging_gpt


class SubmitBody(BaseModel):
    image_filepath: str
    user_prompt: str


app = FastAPI()
origins = [
    # "http://localhost:3000"
    "https://hugginggpt-function-calling-frontend.azurewebsites.net"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {}


@app.get("/example_image_filepaths")
async def get_example_image_filepaths() -> dict[str, list[str]]:
    try:
        example_image_directory = "static/example_images/"
        example_image_filenames = os.listdir(example_image_directory)
        example_image_filepaths = [os.path.join(example_image_directory, example_image_filename)
                                   for example_image_filename in example_image_filenames]
        response = {
            "example_image_filepaths": example_image_filepaths
        }
        return response
    except:
        raise HTTPException(status_code=404, detail="Image files not found.")


@app.get("/example_user_prompts")
async def get_example_user_prompts() -> dict[str, list[str]]:
    try:
        with open("static/example_user_prompts.txt", "r") as f:
            example_user_prompts = f.readlines()
        response = {
            "example_user_prompts": example_user_prompts
        }
        return response
    except:
        raise HTTPException(status_code=404, detail="Text file not found.")


@app.post("/submit")
async def post_submit(submit_body: Annotated[SubmitBody, Body()]) -> dict[str, str]:
    hugging_gpt_instance = hugging_gpt.HuggingGPT(model="gpt-3.5-turbo-0613", is_verbose=True)
    user_content = submit_body.user_prompt + " data: " + submit_body.image_filepath
    assistant_content, used_function_name = hugging_gpt_instance.run(user_content)
    response = {
        "assistant_content": assistant_content,
        "used_function_name": used_function_name
    }
    return response
