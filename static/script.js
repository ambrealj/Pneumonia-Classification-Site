const fileInput = document.getElementById('file-input');
const viewer = document.getElementById('viewer');
const preview = document.getElementById('preview');
const readout = document.getElementById('readout');
const className = document.getElementById('class-name');
const score = document.getElementById('score');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  preview.src = URL.createObjectURL(file);
  viewer.classList.remove('hidden');
  viewer.classList.add('scanning');
  readout.classList.add('hidden');
  className.classList.remove('error');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Request failed (${response.status})`);
    }

    const data = await response.json();

    className.textContent = data.class_name;
    score.textContent = `Confidence: ${data.confidence}%`;
  } catch (err) {
    className.textContent = 'Could not classify image';
    className.classList.add('error');
    score.textContent = err.message;
  } finally {
    viewer.classList.remove('scanning');
    readout.classList.remove('hidden');
  }
});