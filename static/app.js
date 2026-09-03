const askBtn = document.getElementById("askButton");
const answerBox = document.getElementById("answer");
const questionBox = document.getElementById("question");
const presetSelect = document.getElementById("presetQuestion");
const imagePreview = document.getElementById("imagePreview");
const removeImageBtn = document.getElementById("removeImageButton");
const generatedImageContainer = document.getElementById("generatedImageContainer");
const generatedImage = document.getElementById("generatedImage");
const generatedImageTitle = document.getElementById("generatedImageTitle");

let pastedImageDataUrl = "";

function clearGeneratedImage() {
    generatedImage.src = "";
    generatedImageContainer.style.display = "none";
}

function clearPastedImage() {
    pastedImageDataUrl = "";
    imagePreview.src = "";
    imagePreview.style.display = "none";
    removeImageBtn.style.display = "none";
    answerBox.textContent = "Screenshot removed. You can paste a new screenshot or send a text-only question.";
}

function readFileAsDataUrl(file) {
    return new Promise(function (resolve, reject) {
        var reader = new FileReader();
        reader.onload = function () {
            resolve(reader.result);
        };
        reader.onerror = function () {
            reject(reader.error);
        };
        reader.readAsDataURL(file);
    });
}

presetSelect.addEventListener("change", function () {
    if (presetSelect.value !== "") {
        questionBox.value = presetSelect.value;
    }
});

removeImageBtn.addEventListener("click", clearPastedImage);

questionBox.addEventListener("paste", async function (event) {
    var items = event.clipboardData.items;

    for (var i = 0; i < items.length; i++) {
        var item = items[i];
        if (item.type.startsWith("image/")) {
            event.preventDefault();
            var file = item.getAsFile();
            if (!file) {
                return;
            }

            pastedImageDataUrl = await readFileAsDataUrl(file);
            imagePreview.src = pastedImageDataUrl;
            imagePreview.style.display = "block";
            removeImageBtn.style.display = "block";
            answerBox.textContent = "Screenshot pasted successfully. You can now submit it, or add a text question.";
            return;
        }
    }
});

askBtn.addEventListener("click", async function () {
    var question = questionBox.value.trim();

    if (question === "" && pastedImageDataUrl === "") {
        answerBox.textContent = "Please choose a suggested question, type a question, or paste a screenshot first.";
        return;
    }

    askBtn.disabled = true;
    answerBox.textContent = "Thinking...";
    clearGeneratedImage();

    try {
        var res = await fetch("/api/ask-ai", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                environmentalIndicator: "GHG Emissions",
                screenshot: pastedImageDataUrl
            })
        });

        var data = await res.json();

        if (!res.ok) {
            answerBox.textContent =
                "Error: " + (data.error || "Request failed") +
                "\n\nDetail: " + (data.detail || "");
            return;
        }

        answerBox.textContent = data.answer || "No answer returned.";

        if (data.mode === "image" && data.image_base64) {
            generatedImageTitle.textContent = data.title || "AI Generated Image";
            generatedImage.src = "data:image/png;base64," + data.image_base64;
            generatedImageContainer.style.display = "block";
        }
    } catch (err) {
        answerBox.textContent = "Error: " + err.message;
    } finally {
        askBtn.disabled = false;
    }
});