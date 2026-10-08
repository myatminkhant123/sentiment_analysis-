async function analyzeSentiment() {
    const input = document.getElementById('reviewInput');
    const btn = document.getElementById('analyzeBtn');
    const resultArea = document.getElementById('resultArea');
    const resultText = document.getElementById('sentimentResult');
    const resultIcon = document.getElementById('sentimentIcon');
    const resultCard = document.querySelector('.result-card');

    const text = input.value.trim();

    if (!text) {
        alert("Please enter a review first!");
        return;
    }

    // UI Loading State
    btn.classList.add('loading');
    btn.disabled = true;
    resultArea.className = 'result-hidden';

    try {
        const endpoint = (window.location.port === '5000') ? '/predict' : 'http://127.0.0.1:5000/predict';
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review: text }),
        });

        const data = await response.json();

        if (response.ok) {
            // Update UI with Result
            resultText.innerText = data.sentiment;
            resultCard.classList.remove('positive', 'negative');

            if (data.sentiment === 'POSITIVE') {
                resultCard.classList.add('positive');
                resultIcon.innerText = '😊'; // Sparkles for positive
            } else {
                resultCard.classList.add('negative');
                resultIcon.innerText = '😞'; // Disappointed face for negative
            }

            resultArea.className = 'result-visible';
        } else {
            alert("Error: " + (data.error || "Something went wrong"));
        }

    } catch (error) {
        console.error('Error:', error);
        alert("Failed to connect to the server.");
    } finally {
        // Reset Button
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}
