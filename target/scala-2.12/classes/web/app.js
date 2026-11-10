const form = document.getElementById("predictionForm");
const button = document.getElementById("predictButton");
const resultBox = document.getElementById("result");

form.addEventListener("submit", async function (event) {

    // Prevent the page from refreshing
    event.preventDefault();

    resultBox.style.display = "block";
    resultBox.className = "";
    resultBox.textContent = "Processing prediction...";

    button.disabled = true;
    button.textContent = "Predicting...";

    try {

        // Collect all form values
        const formData = new FormData(form);

        // Send data to Scala backend
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
            },
            body: new URLSearchParams(formData).toString()
        });

        const data = await response.json();

        // Handle backend errors
        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        // Display prediction
        const prediction = Number(data.prediction);
        const probability =
            Number(data.positiveProbability) * 100;

        if (prediction === 1) {

            resultBox.className = "success";

            resultBox.innerHTML = `
                <h3>Prediction: Heart Disease Class 1</h3>
                <p>The model classified this patient's data as positive.</p>
                <p>Model score for class 1: ${probability.toFixed(2)}%</p>
            `;

        } else {

            resultBox.className = "success";

            resultBox.innerHTML = `
                <h3>Prediction: Heart Disease Class 0</h3>
                <p>The model classified this patient's data as negative.</p>
                <p>Model score for class 1: ${probability.toFixed(2)}%</p>
            `;
        }

    } catch (error) {

        resultBox.className = "error";
        resultBox.textContent =
            "Error: " + error.message;

    } finally {

        button.disabled = false;
        button.textContent = "Predict";
    }
});