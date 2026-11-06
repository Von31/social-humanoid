// Reactive motion generation + joint (occlusion) denoising schedules (from MAGNet DFoT viz)
(function () {
    'use strict';
    const dfotSection = document.getElementById('denoising');
    if (!dfotSection) return;

    const config = {
        timeSteps: 4,
        blockSize: 40,
        framesPerRow: 40,
    };

    function getDenoiseSteps(mode) {
        // Both modes use the causal-uncertainty (pyramid) schedule: token i starts
        // denoising at step i-1 and takes 3 steps, so nearer tokens are always cleaner (5 rows)
        return 5;
    }

    function getTotalFrames(mode) {
        return getDenoiseSteps(mode) * config.framesPerRow;
    }

    const modeDescriptions = {
        partnerPred: {
            title: 'Reactive Motion Generation',
            text: `Generate the <span class="dfot-description-highlight">humanoid&rsquo;s reaction</span> to the observed human: the robot&rsquo;s future is conditioned on both agents&rsquo; past and the human&rsquo;s motion over the same window.
                <br><br>
                <span class="dfot-description-formula">P(B<sub>t:t+L</sub> | A<sub>0:t+L</sub>, B<sub>0:t−1</sub>)</span>
                <br><br>
                Robot tokens are denoised with <span class="dfot-description-highlight">causal uncertainty</span>: at every step, nearer tokens are less noisy than farther ones, so the response firms up from the present outward. Each human token becomes clean (observed) as its robot token begins denoising.`,
        },
        joint: {
            title: 'Joint Future Generation',
            text: `Used when the human is <span class="dfot-description-highlight">occluded</span>: from the last observed history, the model generates <span class="dfot-description-highlight">both agents&rsquo; futures together</span>, the human&rsquo;s likely motion and the humanoid&rsquo;s coordinated response.
                <br><br>
                <span class="dfot-description-formula">P(A<sub>t:t+L</sub>, B<sub>t:t+L</sub> | A<sub>0:t−1</sub>, B<sub>0:t−1</sub>)</span>
                <br><br>
                Both agents share a noise level at each timestep and follow the same <span class="dfot-description-highlight">causal-uncertainty</span> schedule, nearer frames cleaner than farther ones, which keeps the two futures coordinated.`,
        },
    };

    let state = {
        timeStep: 0,
        isPlaying: true,
        speed: 1,
        mode: 'partnerPred',
    };

    let intervalId = null;

    const playBtn = document.getElementById('playBtn');
    const resetBtn = document.getElementById('resetBtn');
    const speedSlider = document.getElementById('speedSlider');
    const speedValue = document.getElementById('speedValue');
    const frameCount = document.getElementById('frameCount');
    const totalFramesEl = document.getElementById('totalFrames');
    const timelineSlider = document.getElementById('timelineSlider');
    const xAxisLabels = document.getElementById('xAxisLabels');
    const yAxisNumbers = document.getElementById('yAxisNumbers');
    const gridRows = document.getElementById('gridRows');
    const tabPartnerPred = document.getElementById('tabPartnerPred');
    const tabJoint = document.getElementById('tabJoint');
    const descTitle = document.getElementById('dfotDescTitle');
    const descText = document.getElementById('dfotDescText');

    if (!playBtn || !gridRows) return;

    function updateDescription(mode) {
        const desc = modeDescriptions[mode];
        if (!desc) return;
        if (descTitle) descTitle.textContent = desc.title;
        if (descText) descText.innerHTML = desc.text;
    }

    function initXAxisLabels() {
        xAxisLabels.innerHTML = '';
        xAxisLabels.style.marginLeft = '36px';
        xAxisLabels.style.gap = '0';
        for (let t = 0; t < config.timeSteps; t++) {
            const group = document.createElement('div');
            group.className = 'dfot-xaxis-group';
            group.innerHTML = `
                <div class="dfot-xaxis-label">A<sub>${t}</sub></div>
                <div class="dfot-xaxis-label">B<sub>${t}</sub></div>
            `;
            xAxisLabels.appendChild(group);
        }
    }

    function initYAxisNumbers() {
        const denoiseSteps = getDenoiseSteps(state.mode);
        yAxisNumbers.innerHTML = '';
        yAxisNumbers.style.display = 'flex';
        yAxisNumbers.style.paddingTop = '0';
        for (let d = 0; d < denoiseSteps; d++) {
            const num = document.createElement('div');
            num.className = 'dfot-yaxis-number';
            num.id = `yaxis-${d}`;
            num.innerHTML = `<span class="arrow" style="display:none">▶</span>${d}`;
            yAxisNumbers.appendChild(num);
        }
    }

    function initGrid() {
        const denoiseSteps = getDenoiseSteps(state.mode);
        gridRows.innerHTML = '';
        for (let d = 0; d < denoiseSteps; d++) {
            const row = document.createElement('div');
            row.className = 'dfot-row';
            row.id = `row-${d}`;

            for (let t = 0; t < config.timeSteps; t++) {
                const group = document.createElement('div');
                group.className = 'dfot-time-group';

                for (const agent of ['A', 'B']) {
                    const block = document.createElement('div');
                    block.className = 'dfot-block';
                    block.id = `block-${d}-${t}-${agent}`;
                    const canvas = document.createElement('canvas');
                    canvas.width = config.blockSize;
                    canvas.height = config.blockSize;
                    block.appendChild(canvas);
                    group.appendChild(block);
                }

                row.appendChild(group);
            }

            gridRows.appendChild(row);
        }
    }

    // Causal-uncertainty schedule: both agents denoise together, nearer tokens first
    function getNoiseIntensityJoint(denoiseStep, timeIdx, timeStep) {
        if (timeIdx === 0) return 0;
        const tokenNumber = timeIdx - 1;
        const tokenDenoiseStartStep = tokenNumber;
        const denoiseStepsPerToken = 3;
        const framesPerRow = config.framesPerRow;
        const currentDenoiseRow = Math.floor(timeStep / framesPerRow);
        const rowStartFrame = denoiseStep * framesPerRow;
        const isRowActive = timeStep >= rowStartFrame && timeStep < rowStartFrame + framesPerRow;
        const rowProgress = isRowActive ? (timeStep - rowStartFrame) / framesPerRow : 0;

        let targetIntensity;
        if (denoiseStep < tokenDenoiseStartStep) targetIntensity = 1.0;
        else if (denoiseStep < tokenDenoiseStartStep + denoiseStepsPerToken) {
            const stepsInto = denoiseStep - tokenDenoiseStartStep;
            targetIntensity = Math.max(0.0, 1.0 - ((stepsInto + 1) / denoiseStepsPerToken));
        } else targetIntensity = 0;

        let prevIntensity;
        if (denoiseStep === 0) prevIntensity = 1.0;
        else if (denoiseStep - 1 < tokenDenoiseStartStep) prevIntensity = 1.0;
        else if (denoiseStep - 1 < tokenDenoiseStartStep + denoiseStepsPerToken) {
            const stepsInto = (denoiseStep - 1) - tokenDenoiseStartStep;
            prevIntensity = Math.max(0.0, 1.0 - ((stepsInto + 1) / denoiseStepsPerToken));
        } else prevIntensity = 0;

        if (denoiseStep > currentDenoiseRow) {
            if (currentDenoiseRow < tokenDenoiseStartStep) return 1.0;
            if (currentDenoiseRow < tokenDenoiseStartStep + denoiseStepsPerToken) {
                const stepsInto = currentDenoiseRow - tokenDenoiseStartStep;
                const rowPrevIntensity = stepsInto === 0 ? 1.0 : Math.max(0.0, 1.0 - (stepsInto / denoiseStepsPerToken));
                const rowTargetIntensity = Math.max(0.0, 1.0 - ((stepsInto + 1) / denoiseStepsPerToken));
                return rowPrevIntensity + (rowTargetIntensity - rowPrevIntensity) * ((timeStep % framesPerRow) / framesPerRow);
            }
            return 0;
        }
        if (isRowActive) return prevIntensity + (targetIntensity - prevIntensity) * rowProgress;
        return targetIntensity;
    }

    function getNoiseIntensity(denoiseStep, timeIdx, timeStep, agent) {
        if (state.mode === 'joint') {
            return getNoiseIntensityJoint(denoiseStep, timeIdx, timeStep);
        }
        // Reactive: the robot follows the same causal-uncertainty schedule, while each
        // human token is observed (clean) from the step its robot token starts denoising
        if (agent === 'A') return denoiseStep < timeIdx - 1 ? 'empty' : 0;
        return getNoiseIntensityJoint(denoiseStep, timeIdx, timeStep);
    }

    function drawNoise(canvas, intensity) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, config.blockSize, config.blockSize);
        if (intensity < 0.05) return;
        const pixelSize = 3;
        const cols = Math.ceil(config.blockSize / pixelSize);
        const rows = Math.ceil(config.blockSize / pixelSize);
        ctx.globalAlpha = intensity * 0.7;
        for (let i = 0; i < cols * rows; i++) {
            if (Math.random() < intensity) {
                ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.8})`;
                ctx.fillRect((i % cols) * pixelSize, Math.floor(i / cols) * pixelSize, pixelSize, pixelSize);
            }
        }
        ctx.globalAlpha = 1;
    }

    function updateBlock(denoiseStep, timeIdx, agent, timeStep) {
        const block = document.getElementById(`block-${denoiseStep}-${timeIdx}-${agent}`);
        if (!block) return;
        const canvas = block.querySelector('canvas');
        const baseHue = agent === 'A' ? 0 : 210;
        const isContext = timeIdx === 0;
        const noiseIntensity = isContext ? 0 : getNoiseIntensity(denoiseStep, timeIdx, timeStep, agent);

        if (isContext) {
            block.style.backgroundColor = `hsl(${baseHue}, 100%, 35%)`;
            block.style.borderColor = `hsl(${baseHue}, 100%, 50%)`;
            block.style.opacity = '1';
            block.style.borderStyle = 'solid';
            canvas.getContext('2d').clearRect(0, 0, config.blockSize, config.blockSize);
            return;
        }

        if (noiseIntensity === 'empty') {
            block.style.backgroundColor = 'transparent';
            block.style.borderColor = '#334155';
            block.style.borderStyle = 'dashed';
            canvas.getContext('2d').clearRect(0, 0, config.blockSize, config.blockSize);
            return;
        }

        block.style.borderStyle = 'solid';

        if (noiseIntensity <= 0.05) {
            block.style.backgroundColor = `hsl(${baseHue}, 100%, 35%)`;
            block.style.borderColor = `hsl(${baseHue}, 100%, 50%)`;
            block.style.opacity = '1';
            canvas.getContext('2d').clearRect(0, 0, config.blockSize, config.blockSize);
            return;
        }

        const saturation = (1 - noiseIntensity) * 80;
        const lightness = 15 + (1 - noiseIntensity) * 25;
        block.style.backgroundColor = `hsl(${baseHue}, ${saturation}%, ${lightness}%)`;
        block.style.borderColor = `hsl(${baseHue}, ${Math.max(20, saturation)}%, 45%)`;
        drawNoise(canvas, noiseIntensity);
    }

    function update() {
        const denoiseSteps = getDenoiseSteps(state.mode);
        const totalFrames = getTotalFrames(state.mode);
        frameCount.textContent = state.timeStep;
        totalFramesEl.textContent = totalFrames;
        timelineSlider.value = state.timeStep;
        timelineSlider.max = totalFrames - 1;

        for (let d = 0; d < denoiseSteps; d++) {
            const num = document.getElementById(`yaxis-${d}`);
            if (!num) continue;
            const rowStartFrame = d * config.framesPerRow;
            const isActive = state.timeStep >= rowStartFrame && state.timeStep < rowStartFrame + config.framesPerRow;
            num.className = isActive ? 'dfot-yaxis-number active' : 'dfot-yaxis-number';
            num.querySelector('.arrow').style.display = isActive ? 'inline' : 'none';
        }

        for (let d = 0; d < denoiseSteps; d++) {
            const row = document.getElementById(`row-${d}`);
            const rowStartFrame = d * config.framesPerRow;
            const isActive = state.timeStep >= rowStartFrame && state.timeStep < rowStartFrame + config.framesPerRow;
            if (row) row.className = isActive ? 'dfot-row active' : 'dfot-row';

            for (let t = 0; t < config.timeSteps; t++) {
                updateBlock(d, t, 'A', state.timeStep);
                updateBlock(d, t, 'B', state.timeStep);
            }
        }
    }

    function startAnimation() {
        if (intervalId) clearInterval(intervalId);
        if (typeof dfotSectionVisible !== 'undefined' && !dfotSectionVisible) return;
        const isMobile = window.matchMedia('(max-width: 768px)').matches;
        const baseInterval = isMobile ? 100 : 50;
        intervalId = setInterval(() => {
            state.timeStep = (state.timeStep + 1) % getTotalFrames(state.mode);
            update();
        }, baseInterval / state.speed);
    }

    function stopAnimation() {
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
        }
    }

    function switchMode(newMode) {
        if (state.mode === newMode) return;
        const oldSteps = getDenoiseSteps(state.mode);
        state.mode = newMode;
        const newSteps = getDenoiseSteps(state.mode);

        if (tabPartnerPred) tabPartnerPred.classList.toggle('active', newMode === 'partnerPred');
        if (tabJoint) tabJoint.classList.toggle('active', newMode === 'joint');
        updateDescription(newMode);

        state.timeStep = 0;
        if (oldSteps !== newSteps) {
            initYAxisNumbers();
            initGrid();
        }
        update();
        if (state.isPlaying) startAnimation();
    }

    playBtn.addEventListener('click', () => {
        state.isPlaying = !state.isPlaying;
        playBtn.textContent = state.isPlaying ? '⏸ Pause' : '▶ Play';
        playBtn.className = state.isPlaying ? 'dfot-btn dfot-btn-play' : 'dfot-btn dfot-btn-play paused';
        if (state.isPlaying) startAnimation();
        else stopAnimation();
    });

    resetBtn.addEventListener('click', () => {
        state.timeStep = 0;
        update();
    });

    speedSlider.addEventListener('input', (e) => {
        state.speed = parseFloat(e.target.value);
        speedValue.textContent = `${state.speed}x`;
        if (state.isPlaying) startAnimation();
    });

    timelineSlider.addEventListener('input', (e) => {
        state.timeStep = parseInt(e.target.value, 10);
        update();
    });

    if (tabPartnerPred) {
        tabPartnerPred.addEventListener('click', () => switchMode('partnerPred'));
    }
    if (tabJoint) {
        tabJoint.addEventListener('click', () => switchMode('joint'));
    }

    function drawLegendNoise() {
        const canvas = document.getElementById('noisyLegendCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, 16, 16);
        ctx.globalAlpha = 0.7;
        for (let x = 0; x < 16; x += 2) {
            for (let y = 0; y < 16; y += 2) {
                if (Math.random() < 0.5) {
                    ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.8})`;
                    ctx.fillRect(x, y, 2, 2);
                }
            }
        }
        ctx.globalAlpha = 1;
    }

    initXAxisLabels();
    initYAxisNumbers();
    initGrid();
    drawLegendNoise();
    updateDescription(state.mode);

    const isMobile = window.matchMedia('(max-width: 768px)').matches;
    if (isMobile) {
        state.isPlaying = false;
        state.timeStep = getTotalFrames(state.mode) - 1;
        playBtn.textContent = '▶ Play';
        playBtn.className = 'dfot-btn dfot-btn-play paused';
    }

    update();

    let dfotSectionVisible = false;
    const dfotObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            dfotSectionVisible = entry.isIntersecting;
            if (entry.isIntersecting && state.isPlaying && !isMobile) startAnimation();
            else stopAnimation();
        });
    }, { threshold: 0.1 });

    dfotObserver.observe(dfotSection);
})();
