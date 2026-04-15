const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const output = document.getElementById('output');
const earText = document.getElementById('ear');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

// Send frame to Flask
function sendFrame() {
    const ctx = canvas.getContext('2d');
    canvas.width = 640;
    canvas.height = 480;

    ctx.drawImage(video, 0, 0, 640, 480);
    const data = canvas.toDataURL('image/jpeg');

    fetch('/process_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: data })
    })
    .then(res => res.json())
    .then(data => {
        output.src = data.image;
        earText.innerText = data.ear.toFixed(3);
    });
}

// Run every 100ms
setInterval(sendFrame, 100);