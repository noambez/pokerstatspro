from flask import Flask, request, jsonify
from flask_cors import CORS
from montecarlo_ai import PokerSimulator, parse_cards, GeminiPokerAnalyzer, NutsCalculator
import time
from deuces import Evaluator # Already imported

app = Flask(__name__)
CORS(app)

gemini_analyzer = GeminiPokerAnalyzer()
nuts_calculator = NutsCalculator() # Initialize the NutsCalculator

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        hole_cards = parse_cards(' '.join(data['playerCards']))
        community_cards = parse_cards(' '.join(data['communityCards'])) or []
        num_opponents = int(data.get('opponents', 1))

        # Get AI analysis
        analysis = gemini_analyzer.analyze_hand(hole_cards, community_cards, num_opponents)
        
        return jsonify({
            'winProbability': analysis.get('win_probability', 0),
            'strategy': analysis.get('strategy', ''),
            'keyConcepts': analysis.get('key_concepts', []),
            'mistakesToAvoid': analysis.get('mistakes_to_avoid', []),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/montecarlo', methods=['POST'])
def montecarlo():
    try:
        data = request.json
        hole_cards = parse_cards(' '.join(data['playerCards']))
        community_cards = parse_cards(' '.join(data['communityCards'])) or []
        num_opponents = int(data.get('opponents', 1))
        num_simulations = int(data.get('num_simulations', 10000))

        # Run Monte Carlo simulation
        simulator = PokerSimulator(num_opponents)
        results = simulator.run_simulation(hole_cards, community_cards, num_simulations)
        
        return jsonify({
            'winProbability': results['win_probability'],
            'simulations': results['simulations'],
            'executionTime': results['execution_time'],
            'handDistribution': results['hand_distribution'],
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/probability-math', methods=['GET'])
def probability_math():
    try:
        with open('math.html', 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return str(e), 500

@app.route('/assess-hand', methods=['POST'])
def assess_hand():
    try:
        data = request.json
        if not data or 'playerCards' not in data:
            return jsonify({'status': 'error', 'message': 'Invalid request data'}), 400
            
        hole_cards = parse_cards(' '.join(data['playerCards']))
        community_cards = parse_cards(' '.join(data.get('communityCards', ''))) or []
        
        if len(hole_cards) != 2:
            return jsonify({'status': 'error', 'message': 'Select exactly 2 hole cards'}), 400

        evaluator = Evaluator()
        board_deuces = [c.to_deuces() for c in community_cards]
        hand_deuces = [c.to_deuces() for c in hole_cards]
        
        # Evaluate the hand using deuces.Evaluator. This works for any number of community cards (0-5).
        # The evaluator will determine the best 5-card hand from the 7 (or fewer) available cards.
        rank = evaluator.evaluate(board_deuces, hand_deuces)
        rank_class = evaluator.get_rank_class(rank) # Returns 1-9 (1=SF, 9=HC)
        hand_type = evaluator.class_to_string(rank_class) # e.g., "STRAIGHT FLUSH", "PAIR"
        
        # Normalize rank to 0-100 scale. Lower rank value from deuces means stronger hand.
        # The worst possible rank for a 7-card hand is 7462, best is 1.
        # This calculation needs to be robust for pre-flop as well.
        # For pre-flop, the rank might not be a "final" hand type, but deuces still gives a rank.
        # For consistency, we can scale based on the full range of deuces ranks.
        max_deuces_rank = 7462  # Max possible rank value for a 7-card hand
        min_deuces_rank = 1     # Min possible rank value (Royal Flush)
        
        # Calculate strength percentage: higher percentage means stronger hand.
        # If rank is 1 (strongest), strength_percent will be ~100.
        # If rank is 7462 (weakest), strength_percent will be ~0.
        strength_percent = ((max_deuces_rank - rank) / (max_deuces_rank - min_deuces_rank)) * 100
        strength_percent = max(0, min(100, round(strength_percent, 2))) # Clamp between 0 and 100

        # Determine the specific hand class for UI styling (e.g., success, info, warning, danger)
        strength_class = get_hand_strength_class(rank_class)
        
        # Determine the strength title based on the stage of the game or hand type
        strength_title = "Pre-flop Hand Potential" if len(community_cards) == 0 else f"{hand_type}"
        
        # Get a dynamic description based on the hand and community cards
        description = get_hand_description(hand_type, hole_cards, community_cards)

        return jsonify({
            'strengthPercent': strength_percent,
            'strengthLabel': hand_type, # Now using the actual hand type from deuces
            'strengthClass': strength_class,
            'strengthTitle': strength_title,
            'description': description,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Hand analysis failed: {str(e)}'
        }), 500

# Helper function to determine strength class based on deuces hand rank class (1-9)
def get_hand_strength_class(rank_class):
    # Rank classes in deuces range from 1 (straight flush) to 9 (high card)
    if rank_class <= 2:  # Straight Flush, Four of a Kind
        return "success"  # Green (Premium)
    elif rank_class <= 4:  # Full House, Flush
        return "info"     # Blue (Strong)
    elif rank_class <= 6:  # Straight, Three of a Kind
        return "warning"  # Yellow (Average)
    else:  # Two Pair, Pair, High Card
        return "danger"   # Red (Weak)

# Helper function to get description for the hand, dynamically for pre-flop or post-flop
def get_hand_description(hand_type, hole_cards, community_cards):
    if len(community_cards) == 0:
        # Pre-flop description based on hole cards
        ranks_indices = sorted([Card.RANKS.index(c.rank) for c in hole_cards], reverse=True)
        suits = [c.suit for c in hole_cards]
        is_suited = suits[0] == suits[1]
        is_pair = ranks_indices[0] == ranks_indices[1]

        if is_pair:
            if ranks_indices[0] >= Card.RANKS.index('J'): # J, Q, K, A
                return f"A premium pocket pair ({hole_cards[0].rank}{hole_cards[0].suit} {hole_cards[1].rank}{hole_cards[1].suit}). This is an excellent starting hand with high equity and strong potential."
            elif ranks_indices[0] >= Card.RANKS.index('7'): # 7, 8, 9, T
                return f"A medium pocket pair ({hole_cards[0].rank}{hole_cards[0].suit} {hole_cards[1].rank}{hole_cards[1].suit}). A decent opening hand, often played for set mining or small value."
            else:
                return f"A small pocket pair ({hole_cards[0].rank}{hole_cards[0].suit} {hole_cards[1].rank}{hole_cards[1].suit}). Best played for set mining, be cautious with large bets."
        elif is_suited:
            if Card.RANKS.index('A') in ranks_indices and (Card.RANKS.index('K') in ranks_indices or Card.RANKS.index('Q') in ranks_indices):
                return f"Strong suited connectors ({hole_cards[0].rank}{hole_cards[0].suit} {hole_cards[1].rank}{hole_cards[1].suit}). Excellent potential for flushes, straights, and strong pairs."
            elif all(r >= Card.RANKS.index('T') for r in ranks_indices):
                return f"High suited cards ({hole_cards[0].rank}{hole_cards[0].suit} {hole_cards[1].rank}{hole_cards[1].suit}). Good for drawing to flushes and straights, and can make strong pairs."
            else:
                return f"Suited cards ({hole_cards[0].rank}{hole_cards[0].suit} {hole_cards[1].rank}{hole_cards[1].suit}). Primarily played for flush potential and backdoor draws."
        else: # Offsuit
            if Card.RANKS.index('A') in ranks_indices and (Card.RANKS.index('K') in ranks_indices or Card.RANKS.index('Q') in ranks_indices):
                return f"Strong offsuit cards ({hole_cards[0].rank}{hole_cards[0].suit} {hole_cards[1].rank}{hole_cards[1].suit}). Good for making strong pairs, but lacks flush potential."
            elif abs(ranks_indices[0] - ranks_indices[1]) <= 2 and max(ranks_indices) >= Card.RANKS.index('T'):
                return f"Connected offsuit cards ({hole_cards[0].rank}{hole_cards[0].suit} {hole_cards[1].rank}{hole_cards[1].suit}). Some straight potential, but can be tricky to play."
            else:
                return f"Unpaired low cards ({hole_cards[0].rank}{hole_cards[0].suit} {hole_cards[1].rank}{hole_cards[1].suit}). Generally a weak starting hand, often best to fold pre-flop."
    else:
        # Post-flop description based on actual hand type from deuces
        descriptions = {
            "Straight Flush": "A sequence of five consecutive cards of the same suit. This is an extremely rare and powerful hand, virtually unbeatable!",
            "Four of a Kind": "Four cards of the same rank. A monster hand, very difficult to beat.",
            "Full House": "Three cards of one rank and two of another. A very strong hand, often good for extracting maximum value.",
            "Flush": "Five cards of the same suit. A powerful hand, especially if it's the nut flush. Be aware of higher flushes if the board has multiple suits.",
            "Straight": "Five consecutive cards of mixed suits. A strong hand, but can be vulnerable to flushes or higher straights on certain boards.",
            "Three of a Kind": "Three cards of the same rank. A decent hand, often good for value, but be cautious if the board is drawing heavy.",
            "Two Pair": "Two cards of one rank and two of another. A common hand, but be cautious if the board is draw-heavy or if opponents are showing strength.",
            "Pair": "Two cards of the same rank. Often a marginal hand, especially if it's a low pair or there are many overcards on the board.",
            "High Card": "No other hand made, your highest card plays. This is almost always a losing hand; consider folding unless you're bluffing."
        }
        return descriptions.get(hand_type, "Hand type not recognized. This hand's strength is evaluated, but a specific description is not available.")

# No longer needed:
# def evaluate_starting_hand_strength(hole_cards):
#     pass
# def get_preflop_classification(hole_cards, strength_percent):
#     pass

# Helper function to get rank name (used in pre-flop descriptions)
def get_rank_name(rank):
    names = {
        'A': 'Ace', 
        'K': 'King', 
        'Q': 'Queen', 
        'J': 'Jack', 
        'T': '10'
    }
    return names.get(rank, rank)

@app.route('/gemini-recommend', methods=['POST'])
def gemini_recommend():
    try:
        data = request.json
        hole_cards = parse_cards(' '.join(data['playerCards']))
        community_cards = parse_cards(' '.join(data['communityCards'])) or []
        play_style = data.get('playStyle', 'normal')
        num_opponents = int(data.get('opponents', 1))

        # Get play style recommendation from Gemini
        recommendation = gemini_analyzer.get_play_recommendation(
            hole_cards, 
            community_cards, 
            num_opponents, 
            play_style
        )
        
        return jsonify({
            'recommendation': recommendation,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/nuts-calculator', methods=['POST'])
def nuts_calculator_route():
    try:
        data = request.json
        community_cards_str = data.get('communityCards', [])
        
        if not community_cards_str:
            return jsonify({'status': 'error', 'message': 'Please provide community cards.'}), 400

        community_cards = parse_cards(' '.join(community_cards_str))

        if not (3 <= len(community_cards) <= 5):
            return jsonify({'status': 'error', 'message': 'Community cards must be between 3 (flop) and 5 (river) for nuts calculation.'}), 400

        top_hands = nuts_calculator.find_nuts(community_cards)
        
        return jsonify({
            'status': 'success',
            'topHands': top_hands
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
        
if __name__ == '__main__':
    app.run(debug=True, port=5000)
