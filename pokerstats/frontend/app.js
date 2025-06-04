document.addEventListener('DOMContentLoaded', () => {
    initializeCardDropdowns();
    setupTabs();
    setupEventListeners();
    
    // Set up the hand strength tab event listener just once, here
    document.getElementById('hand-strength-tab').addEventListener('shown.bs.tab', function() {
        updateHandStrength();
    });
    // Set up the nuts calculator tab event listener
    document.getElementById('nuts-calculator-tab').addEventListener('shown.bs.tab', function() {
        // Optional: Trigger nuts calculation automatically or set up initial state
        // calculateNuts(); 
    });
});

const SUITS = ['♥', '♦', '♣', '♠'];
const RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
const API_BASE_URL = 'http://localhost:5000';

// Initialize card dropdowns
function initializeCardDropdowns() {
    document.querySelectorAll('.card-select').forEach(select => {
        select.innerHTML = '<option value="">Select Card</option>';
        SUITS.forEach(suit => {
            RANKS.forEach(rank => {
                const card = `${rank}${suit}`;
                select.innerHTML += `<option value="${card}" class="${suit === '♥' || suit === '♦' ? 'text-danger' : ''}">${card}</option>`;
            });
        });
    });
}

// Set up Bootstrap tabs
function setupTabs() {
    const tabElements = document.querySelectorAll('[data-bs-toggle="tab"]');
    tabElements.forEach(tab => new bootstrap.Tab(tab));
    document.getElementById('probability-tab').addEventListener('shown.bs.tab', loadProbabilityMath);
    
    // Add this line to initialize the refresh button when DOM loads
    addRefreshButton();
    
    // Add this event listener to make sure button exists when tab is shown
    document.getElementById('hand-strength-tab').addEventListener('shown.bs.tab', function() {
        updateHandStrength();
        addRefreshButton(); // Ensure refresh button is present
    });
}

// Load probability math content
async function loadProbabilityMath() {
    try {
        const response = await fetch(`${API_BASE_URL}/probability-math`);
        document.getElementById('probability-math-content').innerHTML = await response.text();
    } catch (error) {
        document.getElementById('probability-math-content').innerHTML = `
            <div class="alert alert-danger">Failed to load probability math: ${error.message}</div>`;
    }
}

// Set up event listeners
function setupEventListeners() {
    document.querySelectorAll('.card-select').forEach(select => {
        select.addEventListener('change', validateCardSelections);
        // Also trigger hand strength update when cards change if we're on that tab
        select.addEventListener('change', () => {
            if (document.querySelector('#hand-strength.active')) {
                updateHandStrength();
            }
        });
    });

    document.getElementById('calculate-btn').addEventListener('click', analyzeHand);
    document.getElementById('bluff-btn').addEventListener('click', () => getRecommendation('bluff'));
    document.getElementById('educated-bluff-btn').addEventListener('click', () => getRecommendation('educated_bluff'));
    document.getElementById('normal-play-btn').addEventListener('click', () => getRecommendation('normal'));
    document.getElementById('run-simulation-btn').addEventListener('click', runMonteCarloSimulation);
    document.getElementById('analyze-nuts-btn').addEventListener('click', calculateNuts);
}

// Fixed: Move updateHandStrength to global scope
// Handle hand strength analysis
async function updateHandStrength() {
    const { playerCards, communityCards } = getSelectedCards();
    const loadingDiv = document.getElementById('strength-loading');
    const resultsDiv = document.getElementById('strength-results');
    
    try {
        // Validate input
        if (playerCards.length < 2) {
            document.getElementById('strength-description').innerHTML = 
                '<div class="alert alert-warning">Please select 2 hole cards</div>';
            return;
        }
        
        // Show loading state
        loadingDiv.style.display = 'block';
        resultsDiv.style.display = 'none';
        
        // Convert cards for backend
        const convertedCards = {
            playerCards: playerCards.map(convertToPythonFormat),
            communityCards: communityCards.map(convertToPythonFormat)
        };

        const response = await fetch(`${API_BASE_URL}/assess-hand`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(convertedCards)
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        if (data.status === 'error') throw new Error(data.message);

        // Update UI with results
        loadingDiv.style.display = 'none';
        resultsDiv.style.display = 'block';

        // Update hand display
        const handDisplay = document.getElementById('hand-display');
        handDisplay.innerHTML = playerCards.map(card => `
            <div class="card-symbol ${['♥','♦'].includes(card.slice(-1)) ? 'text-danger' : ''}">
                ${card}
            </div>
        `).join('');

        // Update strength bar
        const strengthBar = document.getElementById('strength-bar');
        strengthBar.style.width = `${data.strengthPercent || 0}%`;
        strengthBar.textContent = data.strengthLabel || 'Unknown';
        strengthBar.className = `progress-bar progress-bar-striped bg-${data.strengthClass || 'secondary'}`;

        // Update description
        const descriptionDiv = document.getElementById('strength-description');
        descriptionDiv.innerHTML = `
            <div class="alert alert-${data.strengthClass}">
                <h5>${data.strengthTitle}</h5>
                <p class="mb-0">${data.description}</p>
            </div>
        `;
    } catch (error) {
        loadingDiv.style.display = 'none';
        resultsDiv.style.display = 'block';
        document.getElementById('strength-description').innerHTML = `
            <div class="alert alert-danger">
                Hand analysis failed: ${error.message}
            </div>
        `;
        console.error('Hand strength error:', error);
    }
}

// Validate card selections
function validateCardSelections() {
    const selectedCards = [];
    document.querySelectorAll('.card-select').forEach(select => {
        if (select.value) {
            if (selectedCards.includes(select.value)) {
                select.value = '';
                showAlert('Duplicate card selected!', 'warning');
            } else {
                selectedCards.push(select.value);
            }
        }
    });
}

// Main analysis function
async function analyzeHand() {
    const { playerCards, communityCards, opponents } = getSelectedCards();
    
    if (playerCards.length < 2) {
        showAlert('Please select 2 hole cards', 'warning');
        return;
    }
    
    try {
        showLoading('results', 'Analyzing hand with AI...');
        
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                playerCards: playerCards.map(convertToPythonFormat),
                communityCards: communityCards.map(convertToPythonFormat),
                opponents: opponents
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'error') throw new Error(data.message);
        
        // Display AI analysis results
        const resultsDiv = document.getElementById('results');
        const winProbability = data.winProbability || 0;
        
        resultsDiv.innerHTML = `
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0">AI Hand Analysis</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card h-100">
                                <div class="card-header">Win Probability</div>
                                <div class="card-body text-center">
                                    <div class="display-4 text-primary">${winProbability.toFixed(2)}%</div>
                                    <p class="text-muted mt-2">Estimated winning chance</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-8">
                            <div class="card h-100">
                                <div class="card-header">Strategic Recommendation</div>
                                <div class="card-body" style="min-height: 200px;">
                                    <div class="strategy-content" style="white-space: pre-wrap">${data.strategy || 'No strategy available'}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">Key Concepts to Apply</div>
                                <div class="card-body">
                                    <ul class="list-group">
                                        ${(data.keyConcepts || []).map(concept => `
                                            <li class="list-group-item d-flex align-items-center">
                                                <span class="badge bg-primary me-2">✓</span>
                                                ${concept}
                                            </li>
                                        `).join('')}
                                        ${data.keyConcepts?.length === 0 ? '<li class="list-group-item text-muted">No key concepts provided</li>' : ''}
                                    </ul>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">Common Mistakes to Avoid</div>
                                <div class="card-body">
                                    <ul class="list-group">
                                        ${(data.mistakesToAvoid || []).map(mistake => `
                                            <li class="list-group-item d-flex align-items-center list-group-item-danger">
                                                <span class="badge bg-danger me-2">✗</span>
                                                ${mistake}
                                            </li>
                                        `).join('')}
                                        ${data.mistakesToAvoid?.length === 0 ? '<li class="list-group-item text-muted">No mistakes listed</li>' : ''}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>`;
            
    } catch (error) {
        showAlert(`Analysis failed: ${error.message}`, 'danger', 'results');
    }
}

// Get AI recommendation
async function getRecommendation(style) {
    const { playerCards, communityCards, opponents } = getSelectedCards();
    
    try {
        showLoading('recommendation', `Generating ${style.replace('_', ' ')} advice...`);
        
        const response = await fetch(`${API_BASE_URL}/gemini-recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                playerCards: playerCards.map(convertToPythonFormat),
                communityCards: communityCards.map(convertToPythonFormat),
                playStyle: style,
                opponents: opponents
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'error') throw new Error(data.message);
        
        const recommendationDiv = document.getElementById('recommendation');
        recommendationDiv.innerHTML = `
            <div class="card border-primary">
                <div class="card-header bg-primary text-white">
                    ${style.replace(/_/g, ' ').toUpperCase()} Recommendation
                </div>
                <div class="card-body">
                    <div class="recommendation-content" style="white-space: pre-wrap">${data.recommendation}</div>
                </div>
            </div>`;
    } catch (error) {
        showAlert(`Recommendation failed: ${error.message}`, 'danger', 'recommendation');
    }
}

// Get selected cards from UI
function getSelectedCards() {
    return {
        playerCards: [
            document.getElementById('player-card-1').value,
            document.getElementById('player-card-2').value
        ].filter(Boolean),
        communityCards: [
            document.getElementById('community-card-1').value,
            document.getElementById('community-card-2').value,
            document.getElementById('community-card-3').value,
            document.getElementById('community-card-4').value,
            document.getElementById('community-card-5').value
        ].filter(Boolean),
        opponents: parseInt(document.getElementById('opponents').value) || 1
    };
}

// Convert card format to Python backend format
function convertToPythonFormat(card) {
    const suitMap = { '♥': 'h', '♦': 'd', '♣': 'c', '♠': 's' };
    let rank = card.slice(0, -1);
    // Convert '10' to 'T' for backend compatibility
    if (rank === '10') rank = 'T';
    return rank + suitMap[card.slice(-1)];
}

// Show loading state
function showLoading(elementId, message) {
    document.getElementById(elementId).innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" style="width: 3rem; height: 3rem;"></div>
            <h5 class="mt-3">${message}</h5>
        </div>`;
}

// Show alert messages
function showAlert(message, type = 'danger', elementId = null) {
    const alert = `
        <div class="alert alert-${type} alert-dismissible fade show">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>`;
    
    if (elementId) {
        document.getElementById(elementId).innerHTML = alert;
    } else {
        const container = document.createElement('div');
        container.innerHTML = alert;
        document.body.prepend(container);
    }
}
// Add this function to your app.js file
function addRefreshButton() {
    // Find the hand-strength tab content
    const strengthTab = document.getElementById('hand-strength');
    
    // Find the first card element or create a container
    let strengthHeader = strengthTab.querySelector('.card-header');
    
    // If there isn't a header yet, this handles the case where we need to add structure
    if (!strengthHeader) {
        // Find the strength results container
        const resultsDiv = document.getElementById('strength-results');
        if (resultsDiv) {
            // First check if there's already a parent card
            let parentCard = resultsDiv.closest('.card');
            if (!parentCard) {
                // Wrap results in a card if not already
                const cardWrapper = document.createElement('div');
                cardWrapper.className = 'card';
                resultsDiv.parentNode.insertBefore(cardWrapper, resultsDiv);
                cardWrapper.appendChild(resultsDiv);
                
                // Add card header
                strengthHeader = document.createElement('div');
                strengthHeader.className = 'card-header d-flex justify-content-between align-items-center';
                strengthHeader.innerHTML = '<h5 class="mb-0">Hand Strength Analysis</h5>';
                cardWrapper.insertBefore(strengthHeader, resultsDiv);
            } else {
                // Get existing header or create one
                strengthHeader = parentCard.querySelector('.card-header');
                if (!strengthHeader) {
                    strengthHeader = document.createElement('div');
                    strengthHeader.className = 'card-header d-flex justify-content-between align-items-center';
                    strengthHeader.innerHTML = '<h5 class="mb-0">Hand Strength Analysis</h5>';
                    parentCard.insertBefore(strengthHeader, resultsDiv);
                }
            }
        }
    }
    
    // Ensure header has flex display for button positioning
    if (strengthHeader && !strengthHeader.classList.contains('d-flex')) {
        strengthHeader.classList.add('d-flex', 'justify-content-between', 'align-items-center');
        
        // Make sure there's a title
        if (!strengthHeader.querySelector('h5')) {
            strengthHeader.innerHTML = '<h5 class="mb-0">Hand Strength Analysis</h5>' + strengthHeader.innerHTML;
        }
    }
    
    // Create refresh button if header exists
    if (strengthHeader) {
        // Check if button already exists
        if (!strengthHeader.querySelector('#refresh-strength-btn')) {
            const refreshBtn = document.createElement('button');
            refreshBtn.id = 'refresh-strength-btn';
            refreshBtn.className = 'btn btn-sm btn-outline-primary';
            refreshBtn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Refresh';
            refreshBtn.addEventListener('click', updateHandStrength);
            
            // Add button to header
            strengthHeader.appendChild(refreshBtn);
        }
    }
}

// Run Monte Carlo simulation
async function runMonteCarloSimulation() {
    const { playerCards, communityCards, opponents } = getSelectedCards();
    const iterations = parseInt(document.getElementById('simulation-runs').value) || 10000;
    
    try {
        showLoading('simulation-results', 'Running Monte Carlo simulation...');
        
        const response = await fetch(`${API_BASE_URL}/montecarlo`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                playerCards: playerCards.map(convertToPythonFormat),
                communityCards: communityCards.map(convertToPythonFormat),
                opponents: opponents,
                num_simulations: iterations
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'error') throw new Error(data.message);
        
        // Safely handle potential missing properties
        const simulations = data.simulations || 0;
        const execTime = data.executionTime || 0;
        const winProb = data.winProbability || 0;
        const handDist = data.handDistribution || {};

        // Build hand distribution table
        const distributionRows = Object.entries(handDist)
            .sort((a, b) => b[1] - a[1])
            .map(([hand, percent]) => `
                <tr>
                    <td>${hand}</td>
                    <td>${percent?.toFixed(2) || 0}%</td>
                </tr>
            `).join('');

        document.getElementById('simulation-results').innerHTML = `
            <div class="card">
                <div class="card-header">Monte Carlo Simulation Results</div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card mb-3">
                                <div class="card-body">
                                    <h5 class="card-title">${winProb.toFixed(2)}%</h5>
                                    <p class="card-text">Win Probability</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card mb-3">
                                <div class="card-body">
                                    <h5 class="card-title">${simulations.toLocaleString()}</h5>
                                    <p class="card-text">Simulations Run</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card mb-3">
                                <div class="card-body">
                                    <h5 class="card-title">${execTime.toFixed(2)}s</h5>
                                    <p class="card-text">Execution Time</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">Hand Distribution</h5>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    <th>Hand Type</th>
                                    <th>Probability</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${distributionRows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>`;
    } catch (error) {
        showAlert(`Simulation failed: ${error.message}`, 'danger', 'simulation-results');
    }
}

// New function to calculate nuts hands
async function calculateNuts() {
    const { communityCards } = getSelectedCards();
    const nutsResultsDiv = document.getElementById('nuts-results');

    if (communityCards.length < 3 || communityCards.length > 5) {
        showAlert('Please select 3, 4, or 5 community cards to calculate the nuts.', 'warning', 'nuts-results');
        return;
    }

    try {
        showLoading('nuts-results', 'Calculating the nuts...');

        const response = await fetch(`${API_BASE_URL}/nuts-calculator`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                communityCards: communityCards.map(convertToPythonFormat)
            })
        });

        const data = await response.json();

        if (data.status === 'error') throw new Error(data.message);

        let nutsHtml = `
            <div class="card">
                <div class="card-header bg-success text-white">
                    <h5>Top 5 Strongest Possible Hands (The Nuts)</h5>
                </div>
                <div class="card-body">
                    <p>Based on the community cards: <strong>${communityCards.join(', ')}</strong></p>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Hole Cards</th>
                                    <th>Best 5-Card Hand</th>
                                    <th>Hand Type</th>
                                </tr>
                            </thead>
                            <tbody>
        `;

        data.topHands.forEach((hand, index) => {
            nutsHtml += `
                <tr>
                    <td>${index + 1}</td>
                    <td><span class="${['♥','♦'].includes(hand.hand_cards[0].slice(-1)) ? 'text-danger' : ''}">${hand.hand_cards[0]}</span> <span class="${['♥','♦'].includes(hand.hand_cards[1].slice(-1)) ? 'text-danger' : ''}">${hand.hand_cards[1]}</span></td>
                    <td>${hand.best_5_card_hand.map(card => `<span class="${['♥','♦'].includes(card.slice(-1)) ? 'text-danger' : ''}">${card}</span>`).join(' ')}</td>
                    <td>${hand.hand_type}</td>
                </tr>
            `;
        });

        nutsHtml += `
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
        nutsResultsDiv.innerHTML = nutsHtml;

    } catch (error) {
        showAlert(`Nuts calculation failed: ${error.message}`, 'danger', 'nuts-results');
    }
}