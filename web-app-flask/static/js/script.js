function uploadFile() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];
  if (file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Failed to upload file');
      }
      return response.blob();
    })
    .then(blob => {
      const resultContainer = document.getElementById('resultContainer');
      resultContainer.innerHTML = ""; //mengosongkan konten sebelum menambahkan hasil deteksi baru
      
      if (blob.type.startsWith('image/')) {
        const imageUrl = URL.createObjectURL(blob);
        resultContainer.innerHTML = `<img src="${imageUrl}" alt="Detected Image" style="min-height: 360px; max-height: 360px;">`;

      } else if (blob.type.startsWith('video/')) {
        const videoUrl = URL.createObjectURL(blob);
        resultContainer.innerHTML = `<video controls autoplay name="media"><source src="static/runs/detect/mp4/result.mp4" style="min-height: 480px; max-height: 480px;"></video>`;
        //resultContainer.innerHTML = ` <video controls="" autoplay="" name="media"><source src="{{ url_for('static', filename='/runs/detect/mp4/video3.mp4') }}" type="video/mp4"></video> `;

      } else {
        throw new Error('Unsupported file type');
      }
    })
    .catch(error => {
      console.error('Error:', error);
      const resultContainer = document.getElementById('resultContainer');
      resultContainer.innerText = "Error: " + error.message;
    });
  }
}
