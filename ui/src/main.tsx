import React, { Suspense } from "react";
import ReactDOM from "react-dom/client";
import { waitForApp } from "./utils/comfyapp.ts";

const App = React.lazy(() =>
  import("./App.tsx").then(({ default: App }) => ({
    default: App,
  })),
);

// hacky fix for chakra add extra className (chakra-ui-light) into body
// and resulted in user unable to copy new nodes into clipboard, because of this check: e.target.className === "litegraph"
// (https://github.com/comfyanonymous/ComfyUI/blob/57926635e8d84ae9eea4a0416cc75e363f5ede45/web/scripts/app.js#L861)
const targetNode = document.body;
const observerConfig = { attributes: true, attributeFilter: ["class"] };
// Callback function to execute when mutations are observed
const callback = function (
  mutationsList: MutationRecord[],
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _observer: MutationObserver,
) {
  // remove color-scheme property from <html> element, this made the checkboxes dark
  const htmlElement = document.documentElement;
  if (htmlElement.style.colorScheme === "dark") {
    // Remove the color-scheme property
    htmlElement.style.removeProperty("color-scheme");
  }

  // remove chakra from <body> class list, this broke the copy node feature
  for (const mutation of mutationsList) {
    if (mutation.type === "attributes" && mutation.attributeName === "class") {
      // remove all chakra classes from body element
      for (const className of targetNode.classList) {
        if (className.includes("chakra")) {
          targetNode.classList.remove(className);
        }
      }
    }
  }
};
const observer = new MutationObserver(callback);
observer.observe(targetNode, observerConfig);

function waitForDocumentBody() {
  return new Promise((resolve) => {
    if (document.body) {
      return resolve(document.body);
    }

    document.addEventListener("DOMContentLoaded", () => {
      resolve(document.body);
    });
  });
}

// wait for document.body to load so that the top menu is loaded and my react component has a place to mount
waitForDocumentBody()
  .then(() => waitForApp())
  .then(() => {
    const topbar = document.createElement("div");
    document.body.append(topbar);
    ReactDOM.createRoot(topbar).render(
      <React.StrictMode>
        <Suspense>
          <App />
        </Suspense>
      </React.StrictMode>,
    );
  });
