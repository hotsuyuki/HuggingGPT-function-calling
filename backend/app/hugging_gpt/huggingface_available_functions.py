import huggingface_hub
import json
import os
import requests
import termcolor


DEBUG_PRINT_COLOR = "yellow"
LOCALHOST = "localhost"


class HuggingfaceAvailableFunctions:
    def __init__(self, huggingface_api_key: str, is_verbose: bool = False) -> None:
        print(termcolor.colored("HuggingfaceAvailableFunctions.__init__()", DEBUG_PRINT_COLOR))
        print()

        self.huggingface_tasks = requests.get("https://huggingface.co/api/tasks").json()
        self.huggingface_inference_client = huggingface_hub.InferenceClient(token=huggingface_api_key)
        self.is_verbose = is_verbose

        with open(os.path.join(os.path.dirname(__file__), "huggingface_available_functions.json"), "r") as f:
            self.available_functions_list = json.load(f)

        if self.is_verbose:
            print(termcolor.colored(f"{self.available_functions_list = }", DEBUG_PRINT_COLOR))
            print()

    def get_available_functions_list(self) -> list:
        print(termcolor.colored("HuggingfaceAvailableFunctions.get_available_functions_list()", DEBUG_PRINT_COLOR))
        print()

        return self.available_functions_list

    def call_function(self, function_name: str, function_arguments: dict, user_content: str) -> str:
        print(termcolor.colored("HuggingfaceAvailableFunctions.call_function()", DEBUG_PRINT_COLOR))
        print()

        function_to_call = getattr(self, function_name.replace("-", "_"))
        function_content = function_to_call(function_name, function_arguments, user_content)
        return function_content

    """
    Computer Vision
    """

    # https://huggingface.co/tasks/image-classification
    def image_classification(self, function_name: str, function_arguments: dict, user_content: str) -> str:
        function_content = \
            json.dumps(self.huggingface_inference_client.image_classification(function_arguments["data"]))
        return function_content

    # https://huggingface.co/tasks/object-detection
    def object_detection(self, function_name: str, function_arguments: dict, user_content: str) -> str:
        recommended_model = self.huggingface_tasks[function_name]["widgetModels"][0]
        function_content = self.huggingface_inference_client.post(
            data=function_arguments["data"],
            model=recommended_model
        ).text
        return function_content

    """
    Multimodal
    """

    # https://huggingface.co/tasks/image-to-text
    def image_to_text(self, function_name: str, function_arguments: dict, user_content: str) -> str:
        function_content = self.huggingface_inference_client.image_to_text(function_arguments["data"])
        return function_content

    # https://huggingface.co/tasks/visual-question-answering
    def visual_question_answering(self, function_name: str, function_arguments: dict, user_content: str) -> str:
        question_mark_index = user_content.find("?")
        json = {
            "inputs": {
                "question": user_content[:question_mark_index + 1] if 0 <= question_mark_index else user_content,
                "image": huggingface_hub._inference._b64_encode(function_arguments["data"])
            }
        }
        recommended_model = self.huggingface_tasks[function_name]["widgetModels"][0]
        function_content = self.huggingface_inference_client.post(
            json=json,
            model=recommended_model
        ).text
        return function_content
