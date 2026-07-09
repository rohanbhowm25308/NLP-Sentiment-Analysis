// ==========================================
// NLP SENTIMENT ANALYSIS
// SCRIPT.JS
// ==========================================

const reviewInput = document.getElementById("review");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const voiceBtn = document.getElementById("voiceBtn");

const loading = document.getElementById("loading");
const result = document.getElementById("result");

const sentiment = document.getElementById("sentiment");
const confidence = document.getElementById("confidence");
const progressBar = document.getElementById("progressBar");
const copyBtn = document.getElementById("copyBtn");

// ==========================================
// ANALYZE BUTTON
// ==========================================

analyzeBtn.addEventListener("click", analyzeReview);

async function analyzeReview() {

    const review = reviewInput.value.trim();

    if (review === "") {

        alert("Please enter a movie review.");

        reviewInput.focus();

        return;

    }

    loading.style.display = "block";

    result.style.display = "none";

    analyzeBtn.disabled = true;

    analyzeBtn.innerHTML = "Analyzing...";

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                review: review
            })

        });

        const data = await response.json();

        sentiment.textContent = data.sentiment;

        confidence.textContent = data.confidence + "%";

        progressBar.style.width = data.confidence + "%";

        result.style.display = "block";

    }

    catch (error) {

        alert("Something went wrong.\nPlease try again.");

        console.error(error);

    }

    finally {

        loading.style.display = "none";

        analyzeBtn.disabled = false;

        analyzeBtn.innerHTML = "Analyze Sentiment";

    }

}
/*=========================================
VOICE RECOGNITION
=========================================*/

const SpeechRecognition =
window.SpeechRecognition ||
window.webkitSpeechRecognition;

if(SpeechRecognition){

    const recognition = new SpeechRecognition();

    recognition.lang = "en-US";

    recognition.interimResults = false;

    recognition.continuous = false;

    voiceBtn.addEventListener("click",()=>{

        recognition.start();

        voiceBtn.classList.add("listening");

    });

    recognition.onresult=(event)=>{

        reviewInput.value = event.results[0][0].transcript;

    };

    recognition.onend=()=>{

        voiceBtn.classList.remove("listening");

    };

}
else{

    voiceBtn.style.display="none";

}

/*=========================================================
AI DATA GRID BACKGROUND
=========================================================*/

const canvas = document.getElementById("bgCanvas");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

resizeCanvas();
window.addEventListener("resize", resizeCanvas);

// =============================================
// PARTICLES
// =============================================

const particles = [];
const particleCount = 250;

class Particle {

    constructor() {

        this.reset();

    }

    reset() {

        this.x = Math.random() * canvas.width;

        this.y = Math.random() * canvas.height;

        this.size = Math.random() * 2 + 1;

        this.speed = Math.random() * 4 + 2;

        this.alpha = Math.random();

    }

    update() {

        this.y -= this.speed;

        if (this.y < 0) {

            this.y = canvas.height;

            this.x = Math.random() * canvas.width;

        }

    }

    draw() {

        ctx.beginPath();

        ctx.fillStyle = "rgba(0,220,255," + this.alpha + ")";

        ctx.shadowColor = "#00d9ff";

        ctx.shadowBlur = 12;

        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);

        ctx.fill();

    }

}

for(let i=0;i<particleCount;i++){

    particles.push(new Particle());

}

// =============================================
// GRID
// =============================================

let offset = 0;

function drawGrid(){

    const horizon = canvas.height * 0.45;

    ctx.strokeStyle="rgba(0,170,255,.18)";

    ctx.lineWidth=1;

    // Horizontal

    for(let i=0;i<50;i++){

        const y=horizon+Math.pow(i,1.5)*8-offset%20;

        ctx.beginPath();

        ctx.moveTo(0,y);

        ctx.lineTo(canvas.width,y);

        ctx.stroke();

    }

    // Vertical Perspective

    const center=canvas.width/2;

    for(let x=-canvas.width;x<canvas.width*2;x+=50){

        ctx.beginPath();

        ctx.moveTo(center,horizon);

        ctx.lineTo(x,canvas.height);

        ctx.stroke();

    }

    offset+= 8;

}

// =============================================
// HORIZON GLOW
// =============================================

function glow(){

    const horizon=canvas.height*0.45;

    const gradient=ctx.createLinearGradient(0,horizon-30,0,horizon+50);

    gradient.addColorStop(0,"rgba(0,255,255,0)");

    gradient.addColorStop(.5,"rgba(0,220,255,.9)");

    gradient.addColorStop(1,"rgba(0,255,255,0)");

    ctx.fillStyle=gradient;

    ctx.fillRect(0,horizon-30,canvas.width,80);

}

// =============================================
// STARS
// =============================================

const stars=[];

for(let i=0;i<250;i++){

    stars.push({

        x:Math.random()*canvas.width,

        y:Math.random()*canvas.height*.45,

        r:Math.random()*2,

        a:Math.random()

    });

}

function drawStars(){

    stars.forEach(s=>{

        ctx.beginPath();

        ctx.fillStyle="rgba(180,240,255,"+s.a+")";

        ctx.arc(s.x,s.y,s.r,0,Math.PI*2);

        ctx.fill();

    });

}

// =============================================
// ANIMATION
// =============================================

function animate(){

    ctx.fillStyle="#020617";

    ctx.fillRect(0,0,canvas.width,canvas.height);

    drawStars();

    glow();

    drawGrid();

    particles.forEach(p=>{

        p.update();

        p.draw();

    });

    requestAnimationFrame(animate);

}

animate();

/*=========================================
COPY RESULT
=========================================*/

copyBtn.addEventListener("click", () => {

    const text =
`Sentiment : ${sentiment.textContent}
Confidence : ${confidence.textContent}`;

    navigator.clipboard.writeText(text);

    copyBtn.innerHTML = " Copied!";

    copyBtn.classList.add("copied");

    setTimeout(() => {

        copyBtn.innerHTML = " Copy Result";

        copyBtn.classList.remove("copied");

    }, 2000);

});
/*=========================================
CLEAR BUTTON
=========================================*/

clearBtn.addEventListener("click", () => {

    // Clear review
    reviewInput.value = "";

    // Reset output
    sentiment.textContent = "-";

    confidence.textContent = "0%";

    progressBar.style.width = "0%";

    // Hide result card
    result.style.display = "none";

    // Focus back on textarea
    reviewInput.focus();

});