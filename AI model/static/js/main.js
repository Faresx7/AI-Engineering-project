document.addEventListener("DOMContentLoaded", () => {
    const body = document.body;
    const chat = document.getElementById("chat");
    const messages = document.getElementById("messages");
    const form = document.getElementById("form");
    const input = document.getElementById("input");
    const send = document.getElementById("send");
    const themeBtn = document.getElementById("themeBtn");

    let generating = false;

    /* =========================
       THEME
    ========================= */
    const sun = `<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"/></svg>`;
    const moon = `<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>`;

    const savedTheme = localStorage.getItem("qwen-theme") || "dark";
    body.dataset.theme = savedTheme;
    themeBtn.innerHTML = savedTheme === "dark" ? sun : moon;

    themeBtn.onclick = () => {
        const dark = body.dataset.theme === "dark";
        const theme = dark ? "light" : "dark";
        body.dataset.theme = theme;
        localStorage.setItem("qwen-theme", theme);
        themeBtn.innerHTML = theme === "dark" ? sun : moon;
    };

    /* =========================
       TEXTAREA AUTO-RESIZE
    ========================= */
    input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 150) + "px";
    });

    input.addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!generating) form.requestSubmit();
        }
    });

    /* =========================
       ADD MESSAGE
    ========================= */
    function addMessage(text, type) {
        const wrapper = document.createElement("div");
        wrapper.className = `wrapper ${type}-wrapper`;

        const msg = document.createElement("div");
        msg.className = `msg ${type}`;
        msg.textContent = text;

        wrapper.appendChild(msg);
        messages.appendChild(wrapper);

        chat.scrollTop = chat.scrollHeight;
        return { wrapper, msg };
    }

    /* =========================
       COPY WITH HTTP FALLBACK
    ========================= */
    function addCopyButton(wrapper) {
        const actions = document.createElement("div");
        actions.className = "actions";

        const button = document.createElement("button");
        button.className = "copy";
        button.type = "button";
        button.textContent = "Copy";

        button.onclick = async () => {
            const text = wrapper.querySelector(".msg").textContent;

            try {
                if (navigator.clipboard && window.isSecureContext) {
                    await navigator.clipboard.writeText(text);
                } else {
                    const textArea = document.createElement("textarea");
                    textArea.value = text;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand("copy");
                    document.body.removeChild(textArea);
                }
                button.textContent = "Copied!";
                setTimeout(() => button.textContent = "Copy", 1500);
            } catch {
                button.textContent = "Failed";
            }
        };

        actions.appendChild(button);
        wrapper.appendChild(actions);
    }

    /* =========================
       CHAT STREAM HANDLING
    ========================= */
    form.addEventListener("submit", async e => {
        e.preventDefault();
        const prompt = input.value.trim();

        if (!prompt || generating) return;

        generating = true;
        send.disabled = true;

        addMessage(prompt, "user");
        input.value = "";
        input.style.height = "auto";

        const bot = addMessage("", "bot");

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt })
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);
            if (!response.body) throw new Error("No response body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });

                if (chunk.includes("data: ") || chunk.trim().startsWith("{")) {
                    const lines = chunk.split("\n");
                    for (let line of lines) {
                        line = line.replace(/^data:\s*/, "").trim();
                        if (!line || line === "[DONE]") continue;
                        try {
                            const parsed = JSON.parse(line);
                            bot.msg.textContent += parsed.response || parsed.text || parsed.content || "";
                        } catch {
                            bot.msg.textContent += line;
                        }
                    }
                } else {
                    bot.msg.textContent += chunk;
                }

                chat.scrollTop = chat.scrollHeight;
            }

            addCopyButton(bot.wrapper);

        } catch (error) {
            console.error(error);
            bot.msg.textContent = "⚠️ Sorry, I couldn't connect to the server.";
        } finally {
            generating = false;
            send.disabled = false;
            input.focus();
        }
    });
});