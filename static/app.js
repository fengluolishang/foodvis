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


// 清除 AI 生成的图片
function clearGeneratedImage() {
    generatedImage.src = "";
    generatedImageContainer.style.display = "none";
}


// 清除用户粘贴的截图
function clearPastedImage() {
    pastedImageDataUrl = "";
    imagePreview.src = "";
    imagePreview.style.display = "none";
    removeImageBtn.style.display = "none";

    answerBox.textContent =
        "Screenshot removed. You can paste a new screenshot or send a text-only question.";
}


// 把图片文件读取成 Data URL
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


// 根据用户问题判断当前询问的是哪个环境指标
function detectEnvironmentalIndicator(question) {

    var q = question.toLowerCase();

    // Water Scarcity 要放在 Water Use 前面判断
    // 避免两个名称出现混淆

    if (q.includes("water scarcity")) {
        return "Water Scarcity";
    }

    if (q.includes("water use")) {
        return "Water Use";
    }

    if (q.includes("ch4")) {
        return "CH4 Emissions";
    }

    if (q.includes("n2o")) {
        return "N2O Emissions";
    }

    if (q.includes("land use")) {
        return "Land Use";
    }

    if (q.includes("eutrophication")) {
        return "Eutrophication";
    }

    if (q.includes("acidification")) {
        return "Acidification";
    }

    if (q.includes("biodiversity")) {
        return "Biodiversity Impact";
    }

    if (q.includes("ghg") || q.includes("greenhouse gas")) {
        return "GHG Emissions";
    }

    // 如果问题中没有写具体指标，
    // 默认使用 GHG Emissions
    return "GHG Emissions";
}


// 选预设问题以后,自动把问题放进输入框
presetSelect.addEventListener("change", function () {

    if (presetSelect.value !== "") {
        questionBox.value = presetSelect.value;
    }

});


// 删除截图按钮
removeImageBtn.addEventListener("click", clearPastedImage);


// 支持直接 Ctrl + V 粘贴截图
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

            answerBox.textContent =
                "Screenshot pasted successfully. You can now submit it, or add a text question.";

            return;
        }
    }
});


// 点击 Ask AI
askBtn.addEventListener("click", async function () {

    var question = questionBox.value.trim();

    // 如果没有文字也没有截图，就不发送
    if (question === "" && pastedImageDataUrl === "") {

        answerBox.textContent =
            "Please choose a suggested question, type a question, or paste a screenshot first.";

        return;
    }


    // 根据问题文字判断环境指标
    var selectedIndicator = detectEnvironmentalIndicator(question);


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

                // 而是根据问题自动判断指标
                environmentalIndicator: selectedIndicator,

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


        // 显示 AI 的文字回答
        answerBox.textContent =
            data.answer || "No answer returned.";


        // 如果是 AI 生成的柱状图，就显示图片
        if (data.mode === "image" && data.image_base64) {

            generatedImageTitle.textContent =
                data.title || "AI Generated Image";

            generatedImage.src =
                "data:image/png;base64," + data.image_base64;

            generatedImageContainer.style.display = "block";
        }


    } catch (err) {

        answerBox.textContent =
            "Error: " + err.message;

    } finally {

        askBtn.disabled = false;
    }
});
