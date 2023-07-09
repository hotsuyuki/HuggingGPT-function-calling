import huggingface_hub
import json
import os
import re
import requests
import termcolor


DEBUG_PRINT_COLOR = "yellow"
LOCALHOST = "localhost"


class HuggingfaceAvailableFunctions:
    def __init__(self, huggingface_api_key: str, is_verbose: bool = False) -> None:
        print(termcolor.colored("HuggingfaceAvailableFunctions.__init__()", DEBUG_PRINT_COLOR))
        self.huggingface_tasks = requests.get("https://huggingface.co/api/tasks").json()
        self.huggingface_inference_client = huggingface_hub.InferenceClient(token=huggingface_api_key)
        self.is_verbose = is_verbose

        if self.is_verbose:
            print(termcolor.colored(f"{self.huggingface_inference_client = }", DEBUG_PRINT_COLOR))
            print()

        with open(os.path.join(os.path.dirname(__file__), "huggingface_available_functions.json"), "r") as f:
            self.available_functions_list = json.load(f)

        if self.is_verbose:
            print(termcolor.colored(f"{self.available_functions_list = }", DEBUG_PRINT_COLOR))
            print()

    def get_available_functions_list(self) -> list:
        print(termcolor.colored("HuggingfaceAvailableFunctions.get_available_functions_list()", DEBUG_PRINT_COLOR))
        return self.available_functions_list

    def call_function(self, function_name: str, function_arguments: dict) -> str:
        print(termcolor.colored("HuggingfaceAvailableFunctions.call_function()", DEBUG_PRINT_COLOR))
        function_to_call = getattr(self, function_name.replace("-", "_"))
        function_content = function_to_call(function_name, self.modify_function_arguments(function_arguments))
        return function_content

    def modify_function_arguments(self, function_arguments: dict) -> dict:
        print(termcolor.colored("HuggingfaceAvailableFunctions.modify_function_arguments()", DEBUG_PRINT_COLOR))
        if not function_arguments.get("data"):
            return function_arguments

        if self.is_verbose:
            print(termcolor.colored(f"{function_arguments = }", DEBUG_PRINT_COLOR))
            print()

        function_arguments["data"] = function_arguments["data"].replace("http://", "https://")
        print(termcolor.colored(f"{function_arguments = }", DEBUG_PRINT_COLOR))

        # e.g. "http://localhost:8000/abc/xyz" --> ":8000/abc/xyz"
        has_localhost_string = (0 <= function_arguments["data"].find(LOCALHOST))
        if has_localhost_string:
            index_after_localhost_string = function_arguments["data"].find(LOCALHOST) + len(LOCALHOST)
            function_arguments["data"] = function_arguments["data"][index_after_localhost_string:]

        if self.is_verbose:
            print(termcolor.colored(f"{has_localhost_string = }", DEBUG_PRINT_COLOR))
            print(termcolor.colored(f"{function_arguments = }", DEBUG_PRINT_COLOR))
            print()

        # e.g. "http://127.0.0.1:8000/abc/xyz" --> ":8000/abc/xyz"
        ip_address_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        ip_address_pattern_match = re.search(ip_address_pattern, function_arguments["data"])
        has_ip_address = (ip_address_pattern_match is not None)
        if has_ip_address:
            index_after_ip_address = ip_address_pattern_match.end()
            function_arguments["data"] = function_arguments["data"][index_after_ip_address:]

        if self.is_verbose:
            print(termcolor.colored(f"{has_ip_address = }", DEBUG_PRINT_COLOR))
            print(termcolor.colored(f"{function_arguments = }", DEBUG_PRINT_COLOR))
            print()

        if has_localhost_string or has_ip_address:
            # e.g. ":8000/abc/xyz" --> "abc/xyz"
            has_port_number = (function_arguments["data"][0] == ":")
            if has_port_number:
                index_after_port_number = function_arguments["data"].find("/") + 1
                function_arguments["data"] = function_arguments["data"][index_after_port_number:]

            if self.is_verbose:
                print(termcolor.colored(f"{has_port_number = }", DEBUG_PRINT_COLOR))
                print(termcolor.colored(f"{function_arguments = }", DEBUG_PRINT_COLOR))
                print()
       
        return function_arguments
   
    """
    Computer Vision
    """

    # https://huggingface.co/tasks/image-classification
    def image_classification(self, function_name: str, function_arguments: dict) -> str:
        pass

    # https://huggingface.co/tasks/object-detection
    def object_detection(self, function_name: str, function_arguments: dict) -> str:
        print(termcolor.colored("HuggingfaceAvailableFunctions.object_detection()", DEBUG_PRINT_COLOR))
        recommended_model = self.huggingface_tasks[function_name]["models"][0]["id"]
        function_content = self.huggingface_inference_client.post(
            data=function_arguments["data"],
            model=recommended_model
        ).text
        return function_content

    # https://huggingface.co/tasks/video-classification
    def video_classification(self, function_name: str, function_arguments: dict) -> str:
        pass

    # https://huggingface.co/tasks/zero-shot-image-classification
    def zero_shot_image_classification(self, function_name: str, function_arguments: dict) -> str:
        pass

    """
    Multimodal
    """

    # https://huggingface.co/tasks/document-question-answering
    def document_question_answering(self, function_name: str, function_arguments: dict) -> str:
        pass

    # https://huggingface.co/tasks/image-to-text
    def image_to_text(self, function_name: str, function_arguments: dict) -> str:
        print(termcolor.colored("HuggingfaceAvailableFunctions.image_to_text()", DEBUG_PRINT_COLOR))
        function_content = self.huggingface_inference_client.image_to_text(function_arguments["data"])
        return function_content

    # https://huggingface.co/tasks/visual-question-answering
    def visual_question_answering(self, function_name: str, function_arguments: dict) -> str:
        pass
