const statusBox = document.getElementById("status");
const listeningIndicator = document.getElementById("listening-indicator");

function setStatus(message) {
    statusBox.textContent = message;
}

async function postJSON(endpoint, body = {}) {
    try {
        setStatus(`Running ${endpoint}...`);

        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
        });

        const data = await response.json();

        setStatus(JSON.stringify(data, null, 2));

    } catch (err) {
        setStatus(`ERROR:\n${err}`);
    }
}

async function startFreeCommunication() {
    const btn = document.getElementById("talkBtn");

    // Disable button during the entire interaction
    btn.disabled = true;
    btn.textContent = "Listening...";

    if (listeningIndicator) listeningIndicator.style.display = "block";
    setStatus("🎙 Listening — speak now. Recording stops automatically after silence.");

    try {
        // Call /audio/chat — blocks until VAD recording + STT + LLM all complete
        const response = await fetch("/audio/chat", { method: "POST" });
        const data = await response.json();

        if (listeningIndicator) listeningIndicator.style.display = "none";

        if (data.status === "success") {
            setStatus(
                `You said: "${data.transcript}"\n\n` +
                `Baymax: "${data.response}"\n\n` +
                `Tools used: ${JSON.stringify(data.tool_calls, null, 2)}`
            );
        } else {
            setStatus(`Error: ${data.message || JSON.stringify(data, null, 2)}`);
        }

    } catch (err) {
        if (listeningIndicator) listeningIndicator.style.display = "none";
        setStatus(`ERROR:\n${err}`);
    } finally {
        btn.disabled = false;
        btn.textContent = "Talk to Baymax";
    }
}

async function runEmotionDetection() {
    try {
        setStatus("Opening live camera stream...");

        // Open camera stream popup window
        const streamWindow = window.open(
            "",
            "BaymaxCamera",
            "width=900,height=700"
        );

        streamWindow.document.write(`
            <html>
            <head>
                <title>Baymax Live Camera</title>
                <style>
                    body {
                        margin: 0;
                        background: black;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        overflow: hidden;
                    }

                    img {
                        width: 100%;
                        height: auto;
                    }

                    h1 {
                        position: absolute;
                        top: 10px;
                        left: 20px;
                        color: white;
                        font-family: Arial;
                    }
                </style>
            </head>
            <body>
                <h1>Baymax Live Camera</h1>
                <img src="http://192.168.10.2:9000/camera/stream" />
            </body>
            </html>
        `);

        setStatus("Running FER + speech...");

        const response = await fetch("/fer/speak", {
            method: "POST"
        });

        const data = await response.json();

        setStatus(JSON.stringify(data, null, 2));

    } catch (err) {
        setStatus(`ERROR:\n${err}`);
    }
}

async function runIntro() {
    await postJSON("/audio/speak", {
        message:
            "Hello everyone. I am Baymax, an interactive multimodal companion robot."
    });
}

async function runClosing() {
    await postJSON("/audio/speak", {
        message:
            "Thank you for watching the Baymax demonstration."
    });
}
