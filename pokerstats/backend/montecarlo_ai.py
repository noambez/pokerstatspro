import random
import time
import multiprocessing
import os
import requests
import re
import json
import logging
from collections import Counter
from deuces import Evaluator, Card as DeucesCard, Deck as DeucesDeck
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('montecarlo_ai')

class Card:
    RANKS = '23456789TJQKA'
    SUITS = 'hdcs'

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    @classmethod
    def from_string(cls, card_str):
        """
        Converts a string like 'Ah' or '10s' into a Card object.
        FIXED: Handles '10' rank correctly by allowing 3-character strings
        and converting '10' to 'T' for internal consistency.
        """
        if len(card_str) < 2 or len(card_str) > 3: # Allow "10h" (3 chars)
            raise ValueError(f"Invalid card string length: {card_str}")

        if len(card_str) == 3: # For "10h"
            rank = card_str[:2].upper() # "10"
            suit = card_str[2].lower()  # "h"
        else: # For "Ah", "Ks", "2c"
            rank = card_str[0].upper()
            suit = card_str[1].lower()

        # Convert '10' to 'T' for internal consistency with RANKS
        if rank == '10':
            rank = 'T'
        
        if rank not in cls.RANKS or suit not in cls.SUITS:
            raise ValueError(f"Invalid rank or suit in card: {card_str}")
        return cls(rank, suit)

    def __repr__(self):
        """
        FIXED: Special handling for 'T' to display as '10' for user readability.
        """
        display_rank = '10' if self.rank == 'T' else self.rank
        return f"{display_rank}{self.suit}"

    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def to_deuces(self):
        """
        Converts the custom Card object to its deuces integer representation.
        FIXED: Ensures the rank is correctly passed to DeucesCard.new (it will already be 'T' if '10').
        """
        return DeucesCard.new(f"{self.rank}{self.suit}")

def parse_cards(card_string):
    """Parse a string of cards into Card objects"""
    if not card_string or card_string.strip() == "None":
        return []
    
    cards = []
    parts = card_string.strip().split()
    
    for part in parts:
        if len(part) >= 2:
            try:
                cards.append(Card.from_string(part))
            except ValueError:
                pass  # Skip invalid cards
    
    return cards


class Deck:
    def __init__(self, excluded_cards=None):
        excluded_cards = excluded_cards or []
        self.cards = [Card(r, s) for r in Card.RANKS for s in Card.SUITS
                      if Card(r, s) not in excluded_cards]
        random.shuffle(self.cards)

    def deal(self, n=1):
        if len(self.cards) < n:
            raise ValueError("Not enough cards in the deck to deal.")
        return [self.cards.pop() for _ in range(n)]


class PokerSimulator:
    # Corrected hand ranks order (Strongest to Weakest)
    HAND_RANKS = [
        "Straight Flush", "Four of a Kind", "Full House",
        "Flush", "Straight", "Three of a Kind",
        "Two Pair", "Pair", "High Card"
    ]

    def __init__(self, num_opponents=1):
        self.num_opponents = num_opponents
        self.evaluator = Evaluator()

    def evaluate_best(self, hole_cards, community_cards):
        board = [c.to_deuces() for c in community_cards]
        hand = [c.to_deuces() for c in hole_cards]
        score = self.evaluator.evaluate(board, hand)
        # Fixed index calculation
        rank_class = self.evaluator.get_rank_class(score) - 1
        return score, self.HAND_RANKS[rank_class]

    def simulate_batch(self, args):
        hole_cards, community_cards, num_sim = args
        wins = 0
        hand_type_counts = Counter() # Renamed from win_counts for clarity, as it counts all occurrences

        for _ in range(num_sim):
            deck = DeucesDeck() 
            
            known_cards_deuces = [c.to_deuces() for c in hole_cards + community_cards]
            deck.cards = [card for card in deck.cards if card not in known_cards_deuces]

            opponents_deuces = []
            for _ in range(self.num_opponents):
                opponents_deuces.append(deck.draw(2))

            needed_community = 5 - len(community_cards)
            
            # Corrected line: Ensure deck.draw() returns a list, even for a single card
            drawn_community_deuces = deck.draw(needed_community)
            if needed_community == 1:
                drawn_community_deuces = [drawn_community_deuces] # Make it a list if only one card was drawn
            elif needed_community == 0:
                drawn_community_deuces = [] # Ensure it's an empty list if no cards are needed

            full_board_deuces = [c.to_deuces() for c in community_cards] + drawn_community_deuces

            player_score = self.evaluator.evaluate(full_board_deuces, [c.to_deuces() for c in hole_cards])
            
            # Count player's hand type occurrence in this simulation
            player_hand_name = self.HAND_RANKS[self.evaluator.get_rank_class(player_score) - 1]
            hand_type_counts[player_hand_name] += 1 

            is_winner = True
            for opp_hand_deuces in opponents_deuces:
                opp_score = self.evaluator.evaluate(full_board_deuces, opp_hand_deuces)
                if opp_score < player_score: # Lower score is better
                    is_winner = False
                    break
            if is_winner:
                wins += 1

        return wins, hand_type_counts # Return the hand type counts

    def run_simulation(self, hole_cards, community_cards, num_simulations=10000):
        start = time.time()
        procs = min(8, multiprocessing.cpu_count() or 4)
        with multiprocessing.Pool(procs) as pool:
            sims_per = num_simulations // procs
            extras = num_simulations % procs
            tasks = [(hole_cards, community_cards, sims_per + (1 if i < extras else 0))
                     for i in range(procs)]
            results = pool.map(self.simulate_batch, tasks)

        total_wins = sum(r[0] for r in results)
        
        # Aggregate hand type counts from all batches
        total_hand_type_counts = Counter()
        for _, wc in results:
            total_hand_type_counts.update(wc)

        win_prob = total_wins / num_simulations * 100 if num_simulations > 0 else 0
        
        # Calculate hand distribution percentages based on total simulations
        # FIXED: Divided by num_simulations instead of total_wins
        win_dist = {h: (total_hand_type_counts[h] / num_simulations * 100) if num_simulations > 0 else 0 for h in self.HAND_RANKS}
        
        return {
            'win_probability': win_prob,
            'simulations': num_simulations,
            'execution_time': time.time() - start,
            'hand_distribution': win_dist
        }


class NutsCalculator:
    def __init__(self):
        self.evaluator = Evaluator()
        self.HAND_RANKS = [
            "Straight Flush", "Four of a Kind", "Full House",
            "Flush", "Straight", "Three of a Kind",
            "Two Pair", "Pair", "High Card"
        ]

    def find_nuts(self, community_cards):
        """
        Calculates the top 5 strongest possible hands (the "nuts") given the community cards.
        
        Args:
            community_cards (list): A list of Card objects representing the community cards.
            
        Returns:
            list: A list of dictionaries, each containing 'hand_cards' (the 2 hole cards),
                  'best_5_card_hand' (the 5-card hand formed), and 'hand_type' (e.g., "Royal Flush").
                  The list is sorted by hand strength (strongest first).
        FIXED: This method has been rewritten to correctly use the deuces.Evaluator
        and itertools.combinations to find the nuts, replacing the non-existent
        'get_five_card_rank_combination' method and fixing card handling.
        """
        if not (3 <= len(community_cards) <= 5):
            raise ValueError("Community cards must be between 3 (flop) and 5 (river).")

        # Convert community cards to deuces integer format
        board_deuces = [c.to_deuces() for c in community_cards]
        
        # Create a full deuces deck
        deck = DeucesDeck()
        
        # Remove the community cards from the deuces deck
        # This ensures that the hole cards we consider are not already on the board.
        # Use a list comprehension to create a new list, as direct removal can be tricky with deuces ints.
        remaining_deck_deuces = [card for card in deck.cards if card not in board_deuces]

        all_possible_hands_info = []

        # Iterate through all possible 2-card hole card combinations from the remaining deuces deck
        for hole_combo_deuces in itertools.combinations(remaining_deck_deuces, 2):
            hole_cards_deuces = list(hole_combo_deuces) # Convert tuple to list for concatenation

            # Combine hole cards with community cards to form a 7-card hand for evaluation
            seven_card_hand_deuces = board_deuces + hole_cards_deuces
            
            # Evaluate the 7-card hand to find its best 5-card rank
            # The evaluator automatically finds the best 5-card hand within the 7 cards
            # and returns its numerical rank.
            rank = self.evaluator.evaluate(board_deuces, hole_cards_deuces)
            
            # Determine the human-readable hand type string (e.g., "STRAIGHT FLUSH", "FULL HOUSE")
            hand_type = self.HAND_RANKS[self.evaluator.get_rank_class(rank) - 1]

            # Find the actual 5-card combination that forms this best hand.
            # 'deuces' gives the rank, but not the specific 5 cards. We need to find them.
            best_5_card_deuces = []
            
            # Iterate through all possible 5-card combinations that can be formed
            # from the 7 cards (2 hole cards + 5 community cards).
            for combo_of_5 in itertools.combinations(seven_card_hand_deuces, 5):
                # Evaluate this specific 5-card combination.
                # When evaluating a 5-card hand, the board argument is an empty list.
                combo_rank = self.evaluator.evaluate(list(combo_of_5), [])
                
                # If this 5-card combination's rank matches the overall best rank
                # of the 7-card hand, then we've found the specific "best 5 cards".
                if combo_rank == rank:
                    best_5_card_deuces = list(combo_of_5)
                    break # Found the specific 5-card hand, no need to check further combinations

            # Convert deuces integer cards back to your custom string format (e.g., 'Ah', '10s')
            # for display in the frontend. We use the Card class's __repr__ for consistent formatting.
            hole_card_strings = [str(Card.from_string(DeucesCard.int_to_str(c))) for c in hole_cards_deuces]
            best_5_card_strings = [str(Card.from_string(DeucesCard.int_to_str(c))) for c in best_5_card_deuces]

            # Store the information about this possible hand
            all_possible_hands_info.append({
                'rank_value': rank, # The numerical rank (lower is better)
                'hand_cards': hole_card_strings, # The 2 hole cards that created this hand
                'best_5_card_hand': best_5_card_strings, # The actual 5 cards forming the best hand
                'hand_type': hand_type # The human-readable type of hand
            })

        # Sort all collected hands by their 'deuces' rank value.
        # The lowest rank value corresponds to the strongest hand.
        all_possible_hands_info.sort(key=lambda x: x['rank_value'])
        
        # Return only the top 5 strongest hands.
        # This provides the "nuts" and the next few strongest possible hands.
        return all_possible_hands_info[:5]


class GeminiPokerAnalyzer:
    # Initialize rate limiter constants
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 1  # Initial backoff in seconds
    RATE_LIMIT_STATUS = 429
    
    def __init__(self, cache_ttl=3600):
        # Get API key from environment with fallback
        self.api_key = os.getenv("GEMINI_API_KEY") or "" 
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds
        
        # Failed request tracking
        self.consecutive_failures = 0
        self.gemini_available = True
        
        logger.info("Gemini Poker Analyzer initialized")
    
    def _calculate_backoff(self, retry_count):
        """Calculate exponential backoff time in seconds"""
        return min(60, self.INITIAL_BACKOFF * (2 ** retry_count))  # Cap at 60 seconds
    
    def _make_api_request(self, payload, max_retries=None):
        """
        Make a request to the Gemini API with retry logic
        
        Args:
            payload: API request payload
            max_retries: Maximum number of retries (defaults to self.MAX_RETRIES)
            
        Returns:
            API response JSON or None on failure
        """
        if max_retries is None:
            max_retries = self.MAX_RETRIES
            
        # Fast-fail if the API has been detected as unavailable
        if not self.gemini_available and self.consecutive_failures > 5:
            logger.warning("Gemini API marked as unavailable, skipping request")
            return None
            
        # Rate limiting - ensure minimum time between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            # For web requests, we shouldn't sleep - just return None to trigger the fallback
            logger.warning("Rate limiting - minimum interval not reached")
            return None
        
        # Just try once for web requests to avoid blocking
        # We'll use the fallback mechanism if this fails
        try:
            # Update last request time
            self.last_request_time = time.time()
            
            # Make the API request
            logger.debug("Making Gemini API request")
            resp = requests.post(
                f"{self.base_url}?key={self.api_key}",
                json=payload,
                timeout=5  # Short timeout to fail quickly
            )
            
            # Handle rate limiting
            if resp.status_code == self.RATE_LIMIT_STATUS:
                logger.warning("Rate limit hit, using fallback")
                self.consecutive_failures += 1
                return None
                
            # Handle other unsuccessful responses
            resp.raise_for_status()
            
            # Success - reset failure counter
            self.consecutive_failures = 0
            self.gemini_available = True
            
            return resp.json()
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                logger.error(f"API request failed with status {status_code}: {str(e)}")
                
                # If a client/server error occurred
                if 400 <= status_code < 600:
                    self.consecutive_failures += 1
                    
                    # After multiple failures, temporarily mark the API as unavailable
                    if self.consecutive_failures > 5:
                        logger.error("Multiple API failures, marking Gemini API as temporarily unavailable")
                        self.gemini_available = False
            else:
                logger.error(f"API connection error: {str(e)}")
                self.consecutive_failures += 1
            
            # For web requests, we need to fail immediately and use the fallback
            return None
    
    def get_play_recommendation(self, hole_cards, community_cards, num_opponents, play_style="normal"):
        """
        Get a play recommendation with caching and error handling.
        FIXED: Removed the early exit for missing API key.
        The _make_api_request method now handles the API key check and fallback.
        """
        # Create a key based on input parameters
        hole_str = " ".join(str(c) for c in hole_cards)
        community_str = " ".join(str(c) for c in community_cards) if community_cards else "None"
        
        # Check if Gemini API is available - quick fail if it's not
        if not self.gemini_available and self.consecutive_failures > 5:
            return self._generate_fallback_recommendation(hole_cards, community_cards, num_opponents, play_style)
        
        # Adjust the prompt based on play style
        style_desc = {
            "bluff": "Be aggressive and make bold bluffing moves regardless of hand strength",
            "educated_bluff": "Strategically bluff when appropriate based on position and board texture",
            "normal": "Play with a balanced, GTO-oriented approach"
        }
        
        style_guidance = style_desc.get(play_style, "Play with a balanced approach")
        
        prompt = (
            f"As a poker coach, give advice for this Texas Hold'em hand:\n\n"
            f"Player Cards: {hole_str}\n"
            f"Community Cards: {community_str}\n"
            f"Opponents: {num_opponents}\n"
            f"Style Guidance: {style_guidance}\n\n"
            f"Provide specific advice for how to play this hand with the given style. "
            f"Include betting strategy, position considerations, and how to react to various opponent actions."
        )
        
        try:
            # Configure Gemini API call for natural language response
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,  # Slightly higher temperature for style variation
                    "topP": 0.9,
                    "topK": 40,
                    "maxOutputTokens": 800  # Reasonable length for advice
                }
            }

            # Make the API request with retry logic
            response_json = self._make_api_request(payload)
            
            # If API request failed completely, return a fallback message
            if not response_json:
                return self._generate_fallback_recommendation(hole_cards, community_cards, num_opponents, play_style)
            
            # Extract text content
            if 'candidates' not in response_json or not response_json['candidates']:
                return self._generate_fallback_recommendation(hole_cards, community_cards, num_opponents, play_style)

            content = response_json['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # Basic cleaning to improve readability
            content = content.replace('```', '').strip()
            
            return content
        except Exception as e:
            logger.error(f"Error getting recommendation: {str(e)}")
            return self._generate_fallback_recommendation(hole_cards, community_cards, num_opponents, play_style)
    
    def _generate_fallback_recommendation(self, hole_cards, community_cards, num_opponents, play_style):
        """Generate a fallback recommendation when API is unavailable"""
        # Generate a simple fallback based on the cards and style
        hand_desc = self._describe_hand(hole_cards, community_cards)
        
        # Adjust based on play style
        if play_style == "bluff":
            return (f"Your hand: {hand_desc}\n\n"
                    f"Bluffing strategy: Since you're aiming to bluff, consider an aggressive betting line "
                    f"regardless of your actual hand strength. Make a confident bet or raise to represent a strong hand.")
        
        elif play_style == "educated_bluff":
            return (f"Your hand: {hand_desc}\n\n"
                    f"Semi-bluff strategy: Look for good bluffing opportunities based on the board texture. "
                    f"If you have drawing potential, a semi-bluff could be effective. Bet when others show weakness.")
        
        else:  # normal play
            if self._is_strong_hand(hole_cards, community_cards):
                return (f"Your hand: {hand_desc}\n\n"
                        f"Recommended strategy: You have a strong hand. Build the pot with consistent bets and consider a check-raise if in position.")
            else:
                return (f"Your hand: {hand_desc}\n\n"
                        f"Recommended strategy: You have a marginal hand. Play cautiously and be ready to fold to significant pressure.")
    
    def _describe_hand(self, hole_cards, community_cards):
        """Generate a simple description of the hand"""
        hole_str = " ".join(str(c) for c in hole_cards)
        community_str = " ".join(str(c) for c in community_cards) if community_cards else "No community cards yet"
        return f"{hole_str} with {community_str}"
    
    def _is_strong_hand(self, hole_cards, community_cards):
        """Very basic hand strength evaluation"""
        # Check for pocket pairs
        if len(hole_cards) == 2 and hole_cards[0].rank == hole_cards[1].rank:
            # High pocket pairs are strong
            if hole_cards[0].rank in 'AKQJT98':
                return True
        
        # Check for high cards
        high_cards = sum(1 for c in hole_cards if c.rank in 'AKQJT')
        if high_cards >= 2:
            return True
        
        # If we have community cards, we should do a more sophisticated evaluation
        # But for this simple fallback, we'll keep it basic
        return False
        
    def analyze_hand(self, hole_cards, community_cards, num_opponents):
        """Analyze a hand using Gemini API with Monte Carlo fallback"""
        # Create simulator first to avoid unbound variable issue
        simulator = PokerSimulator(num_opponents)
        
        try:
            # For analysis, we'll run a Monte Carlo simulation regardless
            # to have some numerical backup
            mc_results = simulator.run_simulation(hole_cards, community_cards, 2000)
            
            # Prepare Gemini prompt for analysis
            hole_str = " ".join(str(c) for c in hole_cards)
            community_str = " ".join(str(c) for c in community_cards) if community_cards else "None"
            
            prompt = (
                f"As a poker expert, analyze this Texas Hold'em hand:\n\n"
                f"Player Cards: {hole_str}\n"
                f"Community Cards: {community_str}\n"
                f"Opponents: {num_opponents}\n\n"
                f"Monte Carlo simulation shows a {mc_results['win_probability']:.2f}% chance of winning.\n\n"
                f"Please provide:\n"
                f"1. A strategic recommendation for playing this hand\n"
                f"2. 3-5 key concepts to apply in this situation\n"
                f"3. 2-3 common mistakes to avoid\n\n"
                f"Format your response without headings or markdown."
            )
            
            # Configure Gemini API call for structured analysis response
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.4,  # Lower temperature for more consistent analysis
                    "topP": 0.9,
                    "topK": 40,
                    "maxOutputTokens": 1000
                }
            }
            
            # Make the API request
            response_json = self._make_api_request(payload)
            
            # If API request failed, use Monte Carlo results for a fallback
            if not response_json or 'candidates' not in response_json or not response_json['candidates']:
                return self._generate_monte_carlo_fallback(mc_results)
            
            # Extract and parse content
            content = response_json['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # Parse the structured response
            strategy, key_concepts, mistakes = self._parse_analysis_response(content)
            
            return {
                'win_probability': mc_results['win_probability'],
                'strategy': strategy,
                'key_concepts': key_concepts,
                'mistakes_to_avoid': mistakes
            }
            
        except Exception as e:
            logger.error(f"Error analyzing hand: {str(e)}")
            # If anything goes wrong, generate a fallback based on Monte Carlo
            # if simulator is not defined yet
            if not 'simulator' in locals():
                simulator = PokerSimulator(num_opponents)
            mc_results = simulator.run_simulation(hole_cards, community_cards, 1000)
            return self._generate_monte_carlo_fallback(mc_results, str(e))
    
    def _parse_analysis_response(self, content):
        """Parse the AI response into structured data"""
        # Default values
        strategy = "Strategy not available."
        key_concepts = []
        mistakes = []
        
        try:
            # Very basic parsing - split by common heading patterns
            parts = re.split(r'\n\s*\d+[\.\)]\s+|\n\s*-\s+|\n\n+', content)
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 1:
                strategy = parts[0]
            
            # Try to extract key concepts and mistakes
            remaining_parts = parts[1:] if len(parts) > 1 else []
            
            # Identify concepts vs mistakes based on keywords
            for part in remaining_parts:
                lower_part = part.lower()
                if any(word in lower_part for word in ['avoid', 'mistake', 'error', 'wrong', 'don\'t']):
                    mistakes.append(part)
                else:
                    key_concepts.append(part)
            
            # Limit the number of items
            key_concepts = key_concepts[:5]
            mistakes = mistakes[:3]
            
        except Exception as e:
            logger.error(f"Error parsing analysis: {str(e)}")
            # Return the full content as strategy if parsing fails
            strategy = content
        
        return strategy, key_concepts, mistakes
    
    def _generate_monte_carlo_fallback(self, mc_results, error_msg=None):
        """Generate a fallback analysis based on Monte Carlo results"""
        win_prob = mc_results['win_probability']
        
        # Generate a basic strategy based on win probability
        strategy = self._generate_strategy_from_win_probability(win_prob)
        
        # Generate key concepts
        key_concepts = self._generate_key_concepts(win_prob, mc_results.get('hand_distribution', {}))
        
        # Generate mistakes to avoid
        mistakes = self._generate_mistakes_to_avoid(win_prob)
        
        # Add error message if provided
        if error_msg:
            strategy = f"{strategy}\n\nNote: AI analysis unavailable ({error_msg}). Using statistical analysis only."
        
        return {
            'win_probability': win_prob,
            'strategy': strategy,
            'key_concepts': key_concepts,
            'mistakes_to_avoid': mistakes
        }
    
    def _generate_strategy_from_win_probability(self, win_prob):
        """Generate a basic strategy based on win probability"""
        if win_prob >= 70:
            return "You have a very strong hand. Focus on maximizing value by building the pot. Consider slow-playing if the board is draw-heavy to trap opponents."
        elif win_prob >= 50:
            return "You have a strong hand that's likely ahead. Bet for value and protect against draws. Be prepared to call reasonable raises."
        elif win_prob >= 30:
            return "You have a medium-strength hand. Play cautiously and be aware of the board texture. Consider check-calling to control the pot size."
        else:
            return "Your hand is relatively weak. Play defensively and be ready to fold to significant pressure. Look for cheap opportunities to see more cards if you have drawing potential."
    
    def _generate_key_concepts(self, win_prob, hand_distribution):
        """Generate key concepts based on win probability and hand distribution"""
        concepts = []
        
        # Concept based on win probability
        if win_prob >= 60:
            concepts.append("Value betting is key - your hand is strong enough to bet for value on most runouts")
        elif win_prob >= 40:
            concepts.append("Pot control - manage the pot size based on your relative hand strength")
        else:
            concepts.append("Position is crucial - play more hands in late position and fewer from early position")
        
        # Concept about odds
        concepts.append(f"Win probability of {win_prob:.1f}% means you should be {'aggressive' if win_prob > 50 else 'cautious'}")
        
        # Add concepts about drawing if relevant
        if 30 <= win_prob <= 60:
            concepts.append("Consider the pot odds when chasing draws - only continue if the potential payout justifies the cost")
        
        # Add concept about opponents
        concepts.append("Adjust your play based on opponent tendencies - tight players will have stronger ranges")
        
        return concepts
    
    def _generate_mistakes_to_avoid(self, win_prob):
        """Generate common mistakes to avoid based on win probability"""
        mistakes = []
        
        if win_prob >= 70:
            mistakes.append("Don't slow-play too much - build the pot with strong hands")
            mistakes.append("Avoid getting too tricky - straightforward play often yields better results with strong hands")
        elif win_prob >= 50:
            mistakes.append("Don't overvalue your hand - be prepared to fold if the board or action suggests you're behind")
            mistakes.append("Avoid playing too passively - protect your equity with appropriate betting")
        elif win_prob >= 30:
            mistakes.append("Don't chase draws without proper odds - calculate your equity before calling")
            mistakes.append("Avoid bluffing too much - middle-strength hands are better for showdown value")
        else:
            mistakes.append("Don't invest too many chips with a weak hand - minimize losses when behind")
            mistakes.append("Avoid calling too much out of curiosity - fold when the math doesn't support continuing")
        
        return mistakes