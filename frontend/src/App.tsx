import axios from "axios";
import React from "react";
import "./App.css";

type ResponseJsonType = {
  assistant_content: string;
  used_function_name: string;
};

function App() {
  // const serverUrl = "http://localhost:8000";
  const serverUrl = "https://hugginggpt-function-calling-backend.azurewebsites.net";

  const [exampleImageUrls, setExampleImageUrls] = React.useState<string[]>([]);
  const [submitImageUrl, setSubmitImageUrl] = React.useState<string>("");

  const [exampleUserPrompts, setExampleUserPrompts] = React.useState<string[]>([]);
  const [submitUserPrompt, setSubmitUserPrompt] = React.useState<string>("");

  const [isWaitingResponseJson, setIsWaitingResponseJson] = React.useState<boolean>(false);
  const [responseJson, setResponseJson] = React.useState<ResponseJsonType>();

  React.useEffect(() => {
    axios.get(serverUrl + "/example_image_filepaths").then((response) => {
      const exampleImageUrls = response.data.example_image_filepaths.map((example_image_filepath: string) => {
        return serverUrl + "/" + example_image_filepath;
      });
      setExampleImageUrls(exampleImageUrls);
      if (0 < exampleImageUrls.length) {
        setSubmitImageUrl(exampleImageUrls[0]);
      }
    });

    axios.get(serverUrl + "/example_user_prompts").then((response) => {
      setExampleUserPrompts(response.data.example_user_prompts);
    });
  }, []);

  const handleSubmit = () => {
    setIsWaitingResponseJson(true);

    const submitImageFilepath = submitImageUrl.replace(serverUrl + "/", "");
    const submitData = {
      image_filepath: submitImageFilepath,
      user_prompt: submitUserPrompt
    };

    const submitConfig = {
      headers: {
        "Content-Type": "application/json"
      }
    };

    axios.post(serverUrl + "/submit", submitData, submitConfig).then((response) => {
      setResponseJson(response.data);
      setIsWaitingResponseJson(false);
    });
  }

  const showResponseJson = () => {
    if (isWaitingResponseJson) {
      return (
        <div>
          <p>Waiting response...</p>
        </div>
      );
    }

    if (responseJson) {
      const splitTextByLF = (text: string) => {
        return text.split("\n").map((split_text: string) => {
          return (<div>{split_text} <br /></div>);
        })
      }
      return (
        <div>
          <p style={{ fontWeight: "bold" }}>Answer:</p>
          <p>{splitTextByLF(responseJson.assistant_content)}</p>
          <p style={{ fontWeight: "bold" }}>Used function:</p>
          <p>{splitTextByLF(responseJson.used_function_name)}</p>
        </div>
      );
    }
  }

  return (
    <>
      <div className="App-header">
        <h1>HuggingGPT-function-calling</h1>
      </div>

      <div className="App-body">
        <div className="App-side-margin">
        </div>

        <div className="App-each-step">
          <div>Step 1. Select an example image.</div>
          <select onChange={(event) => setSubmitImageUrl(event.target.value)}>
            {exampleImageUrls.map((exampleImageUrl: string, index: number) => {
              const exampleImageFilename = exampleImageUrl.split("/").pop();
              return (<option key={index} value={exampleImageUrl}>{exampleImageFilename}</option>);
            })}
          </select>
          <br />
          <img src={submitImageUrl} width={"80%"} alt={submitImageUrl} />
        </div>

        <div className="App-each-step">
          <div>Step 2. Select an example prompt or input manually.</div>
          <select onChange={(event) => setSubmitUserPrompt(event.target.value)}>
            <option disabled={true} selected={true}>Select an example prompt...</option>
            {exampleUserPrompts.map((exampleUserPrompt: string, index: number) => {
              return (<option key={index} value={exampleUserPrompt}>{exampleUserPrompt}</option>);
            })}
          </select>
          <br />
          <textarea value={submitUserPrompt} onChange={(event) => setSubmitUserPrompt(event.target.value)} rows={4} cols={40} />
          <button onClick={handleSubmit}>Send</button>
          <br />
          {showResponseJson()}
        </div>

        <div className="App-side-margin"></div>
      </div>
    </>
  );
}

export default App;
