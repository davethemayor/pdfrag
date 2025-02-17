let history = [];

async function sendMessage() {
  const userInput = document.getElementById("user-input").value;
  if (userInput.trim() === "") return; // Prevent sending empty messages

  const sendButton = document.querySelector(".send-button");
  sendButton.disabled = true; // Disable the button
  sendButton.innerHTML = '<div class="spinner"></div>'; // Show the spinner

  const response = await fetch("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_input: userInput, history: history }),
  });
  const data = await response.json();

  const chatBox = document.getElementById("chat-box");
  if (chatBox.classList.contains("hidden")) {
    chatBox.classList.remove("hidden");
  }
  const userMessage = `<div class="message user">${userInput}</div>`;
  const assistantMessage = `<div class="message assistant">${marked.parse(
    data.response
  )}</div>`;

  chatBox.innerHTML += userMessage + assistantMessage;
  document.getElementById("user-input").value = "";

  // Scroll to the bottom of the chat box
  chatBox.scrollTop = chatBox.scrollHeight;

  // Update the history
  history.push({ role: "user", content: userInput });
  history.push({ role: "assistant", content: data.response });

  sendButton.disabled = false;
  sendButton.innerHTML = "&#10148;";
}

document
  .getElementById("user-input")
  .addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });
