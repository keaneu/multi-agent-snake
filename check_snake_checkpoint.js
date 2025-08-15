// Check snake game checkpoint data
console.log("🐍 Snake Game Checkpoint Data:");

// Get checkpoint from localStorage
const checkpoint = localStorage.getItem('snake_duel_checkpoint');
if (checkpoint) {
    try {
        const data = JSON.parse(checkpoint);
        console.log("📊 Current Scores:");
        console.log(`  Snake A: ${data.A || 0} points`);
        console.log(`  Snake B: ${data.B || 0} points`);
        
        // Calculate statistics
        const total = (data.A || 0) + (data.B || 0);
        const leader = data.A > data.B ? 'A' : data.B > data.A ? 'B' : 'Tie';
        const margin = Math.abs((data.A || 0) - (data.B || 0));
        
        console.log("📈 Game Statistics:");
        console.log(`  Total Points: ${total}`);
        console.log(`  Leader: Snake ${leader}`);
        if (leader !== 'Tie') {
            console.log(`  Margin: ${margin} points`);
        }
        
        // Export data for visualization
        window.snakeData = data;
        
    } catch (e) {
        console.error("❌ Error parsing checkpoint:", e);
    }
} else {
    console.log("⚠️ No checkpoint data found");
    window.snakeData = {A: 0, B: 0};
}

// Function to create simple visualization
function visualizeResults() {
    const data = window.snakeData || {A: 0, B: 0};
    
    // Create a simple text-based chart
    const maxScore = Math.max(data.A, data.B, 1);
    const barA = '█'.repeat(Math.max(1, Math.floor((data.A / maxScore) * 20)));
    const barB = '█'.repeat(Math.max(1, Math.floor((data.B / maxScore) * 20)));
    
    console.log("\n🎮 Score Visualization:");
    console.log("┌─────────────────────────┐");
    console.log(`│ Snake A: ${data.A.toString().padStart(5)} ${barA}`);
    console.log(`│ Snake B: ${data.B.toString().padStart(5)} ${barB}`);
    console.log("└─────────────────────────┘");
    
    return data;
}

// Run visualization
const results = visualizeResults();

// Create HTML visualization
function createHTMLVisualization() {
    const data = window.snakeData || {A: 0, B: 0};
    const total = data.A + data.B;
    const percentA = total > 0 ? (data.A / total * 100).toFixed(1) : 50;
    const percentB = total > 0 ? (data.B / total * 100).toFixed(1) : 50;
    
    const html = `
    <div style="font-family: Arial, sans-serif; padding: 20px; background: #1a1a1a; color: #fff; border-radius: 10px; margin: 20px;">
        <h2 style="color: #4CAF50; text-align: center;">🐍 Snake Duel Results</h2>
        
        <div style="display: flex; justify-content: space-around; margin: 20px 0;">
            <div style="text-align: center;">
                <h3 style="color: #2196F3;">Snake A</h3>
                <div style="font-size: 2em; font-weight: bold;">${data.A}</div>
                <div style="color: #888;">${percentA}%</div>
            </div>
            <div style="text-align: center; color: #FF9800;">
                <h3>VS</h3>
                <div style="font-size: 1.5em;">⚡</div>
            </div>
            <div style="text-align: center;">
                <h3 style="color: #F44336;">Snake B</h3>
                <div style="font-size: 2em; font-weight: bold;">${data.B}</div>
                <div style="color: #888;">${percentB}%</div>
            </div>
        </div>
        
        <div style="background: #333; border-radius: 10px; overflow: hidden; margin: 20px 0;">
            <div style="display: flex; height: 30px;">
                <div style="background: #2196F3; width: ${percentA}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                    ${percentA > 15 ? 'A' : ''}
                </div>
                <div style="background: #F44336; width: ${percentB}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                    ${percentB > 15 ? 'B' : ''}
                </div>
            </div>
        </div>
        
        <div style="text-align: center; color: #888; font-size: 0.9em;">
            Total Games Played: Autonomous AI vs AI Battle<br>
            Multi-Agent System Performance
        </div>
    </div>`;
    
    // Create and append visualization
    const container = document.createElement('div');
    container.innerHTML = html;
    document.body.appendChild(container);
    
    return html;
}

// Export functions for manual use
window.visualizeResults = visualizeResults;
window.createHTMLVisualization = createHTMLVisualization;

console.log("\n✅ Visualization ready! Run createHTMLVisualization() to show results on page.");