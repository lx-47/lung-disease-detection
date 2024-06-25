document.addEventListener('DOMContentLoaded', function() {
    const uploadInput = document.getElementById('uploadInput');
    const uploadButton = document.getElementById('uploadButton');
    const imageContainer = document.getElementById('imageContainer');
    const spinnerContainer = document.getElementById('spinnerContainer');
    const dropZone = document.getElementById('dropZone');
    const apiUrl = 'http://localhost:8000/predict'; 

    // Trigger file input when upload button is clicked
    uploadButton.addEventListener('click', function() {
        uploadInput.click();
    });

    // Handle file upload via file input
    uploadInput.addEventListener('change', function() {
        const file = uploadInput.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });

    // Handle file upload via drag and drop
    dropZone.addEventListener('dragover', function(event) {
        event.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', function(event) {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', function(event) {
        event.preventDefault();
        dropZone.classList.remove('dragover');
        const file = event.dataTransfer.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });

    // Function to handle file upload and prediction
    function handleFileUpload(file) {
        imageContainer.innerHTML = '';
        spinnerContainer.innerHTML = '<div class="spinner"></div>';

        // Create a FormData object to send the file to the server
        const formData = new FormData();
        formData.append('image', file);

        fetch(apiUrl, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Clear spinner
            spinnerContainer.innerHTML = '';

            // Check if prediction is successful
            if (data.prediction && data.confidence) {
                // Create image element
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                img.classList.add('uploaded-image');

                // Create prediction element
                const predictionElement = document.createElement('div');
                predictionElement.classList.add('prediction-text');
                predictionElement.innerHTML = `<p>Prediction: ${data.prediction}</p><p>Confidence: ${data.confidence.toFixed(2)}%</p>`;

                // Append image and prediction elements to the container
                imageContainer.appendChild(img);
                imageContainer.appendChild(predictionElement);
            } else {
                // Display an error message
                imageContainer.innerHTML = 'Error predicting. Please try again.';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            // Clear spinner and display an error message
            spinnerContainer.innerHTML = '';
            imageContainer.innerHTML = 'Error predicting. Please try again.';
        });
    }
});
