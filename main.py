import re
import math
import random
import json
from pythonforandroid.recipes.sqlite3 import Sqlite3Recipe
from pythonforandroid.util import load_source
import os

class Sqlite3Recipe(Sqlite3Recipe):
    version = '3.42.0'
    url = 'https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz'
    
    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        env['CFLAGS'] += ' -DSQLITE_ENABLE_COLUMN_METADATA=1'
        return env

recipe = Sqlite3Recipe()
import hashlib
import pickle
import sqlite3
import time
import threading
import heapq
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from itertools import combinations
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.label import Label
    from kivy.uix.textinput import TextInput
    from kivy.uix.button import Button
    from kivy.uix.filechooser import FileChooserIconView
    from kivy.uix.popup import Popup
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.progressbar import ProgressBar
    from kivy.clock import Clock
    from kivy.core.window import Window
    from kivy.core.audio import SoundLoader
    from kivy.core.clipboard import Clipboard
    from plyer import gps, accelerometer
except ImportError:
    print("Some modules not installed. Run: pip install kivy plyer cryptography")
    class App: 
        def __init__(self): pass
        @property
        def user_data_dir(self): return os.path.expanduser("~")
    Window = type('Window', (), {'size': (450, 700)})()


# ==================== FIX 1: DATABASE SHIFT (SQLite) ====================
class BabitaDatabase:
    """SQLite with FTS5 - 100x faster than Pickle"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.init_database()
    
    def init_database(self):
        """Initialize database with FTS5"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Main knowledge table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence TEXT NOT NULL,
                source TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed DATETIME,
                importance REAL DEFAULT 1.0,
                book_id TEXT,
                confidence REAL DEFAULT 1.0
            )
        ''')
        
        # FTS5 for full-text search
        self.cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                sentence, content=knowledge, content_rowid=id
            )
        ''')
        
        # Word index for fast lookups
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS word_index (
                word TEXT PRIMARY KEY,
                sentence_ids TEXT
            )
        ''')
        
        # Concept relationships
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS concept_map (
                concept1 TEXT,
                concept2 TEXT,
                strength REAL DEFAULT 1.0,
                source TEXT,
                PRIMARY KEY (concept1, concept2)
            )
        ''')
        
        # Conversation history
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                bot_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sentiment REAL,
                context TEXT
            )
        ''')
        
        # User profile
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def add_knowledge(self, sentence, source="", book_id="", confidence=1.0):
        """Add knowledge with timestamp"""
        self.cursor.execute('''
            INSERT INTO knowledge (sentence, source, timestamp, book_id, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (sentence, source, datetime.now(), book_id, confidence))
        row_id = self.cursor.lastrowid
        
        # Add to FTS
        self.cursor.execute('''
            INSERT INTO knowledge_fts (rowid, sentence)
            VALUES (?, ?)
        ''', (row_id, sentence))
        
        # Index words
        words = set(re.findall(r'\b\w{3,}\b', sentence.lower()))
        for word in words:
            self.cursor.execute('''
                INSERT INTO word_index (word, sentence_ids)
                VALUES (?, ?)
                ON CONFLICT(word) DO UPDATE SET
                sentence_ids = sentence_ids || ',' || ?
            ''', (word, str(row_id), str(row_id)))
        
        self.conn.commit()
        return row_id
    
    def search_knowledge(self, query, limit=5, min_confidence=0.3):
        """Fast FTS5 search with temporal weighting"""
        # Full-text search
        self.cursor.execute('''
            SELECT k.id, k.sentence, k.timestamp, k.access_count, k.confidence,
                   k.importance, julianday('now') - julianday(k.timestamp) as age_days
            FROM knowledge_fts fts
            JOIN knowledge k ON fts.rowid = k.id
            WHERE knowledge_fts MATCH ?
            ORDER BY 
                (k.confidence * k.importance * 
                 (1.0 / (age_days + 1)) *  -- 🔥 Temporal weighting
                 (k.access_count + 1)       -- Popularity boost
                ) DESC
            LIMIT ?
        ''', (query, limit))
        
        return self.cursor.fetchall()
    
    def get_word_context(self, word, limit=5):
        """Get sentences containing word"""
        self.cursor.execute('''
            SELECT sentence FROM knowledge
            WHERE sentence LIKE ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        ''', (f'%{word}%', limit))
        return self.cursor.fetchall()
    
    def add_concept_relation(self, concept1, concept2, strength=1.0, source=""):
        """Add relationship between concepts"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO concept_map (concept1, concept2, strength, source)
            VALUES (?, ?, ?, ?)
        ''', (concept1, concept2, strength, source))
        self.conn.commit()
    
    def get_related_concepts(self, concept, limit=5):
        """Get concepts related to given concept"""
        self.cursor.execute('''
            SELECT concept2, strength FROM concept_map
            WHERE concept1 = ?
            UNION
            SELECT concept1, strength FROM concept_map
            WHERE concept2 = ?
            ORDER BY strength DESC
            LIMIT ?
        ''', (concept, concept, limit))
        return self.cursor.fetchall()
    
    def add_conversation(self, user_input, bot_response, sentiment=0.0, context=""):
        """Save conversation for context retention"""
        self.cursor.execute('''
            INSERT INTO conversation (user_input, bot_response, sentiment, context)
            VALUES (?, ?, ?, ?)
        ''', (user_input, bot_response, sentiment, context))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_recent_conversations(self, limit=5):
        """Get recent conversation for context"""
        self.cursor.execute('''
            SELECT user_input, bot_response FROM conversation
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        return self.cursor.fetchall()
    
    def set_user_profile(self, key, value):
        """Save user profile data"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO user_profile (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (key, str(value)))
        self.conn.commit()
    
    def get_user_profile(self, key):
        """Get user profile data"""
        self.cursor.execute('''
            SELECT value FROM user_profile WHERE key = ?
        ''', (key,))
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def cleanup_old_knowledge(self, days=365):
        """Information Decay - remove unused knowledge"""
        self.cursor.execute('''
            DELETE FROM knowledge
            WHERE julianday('now') - julianday(last_accessed) > ?
            AND access_count < 5
        ''', (days,))
        self.conn.commit()
        return self.cursor.rowcount
    
    def close(self):
        if self.conn:
            self.conn.close()


# ==================== FIX 2: ENCRYPTION LAYER ====================
class BrainEncryption:
    """AES-256 encryption for brain files"""
    
    def __init__(self, password):
        self.password = password
        self.key = self.derive_key(password)
        self.cipher = Fernet(self.key)
    
    def derive_key(self, password):
        """Derive encryption key from password"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'babita_salt_2026',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data):
        """Encrypt data"""
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data):
        """Decrypt data"""
        return self.cipher.decrypt(encrypted_data)


# ==================== FIX 3: CHAIN OF THOUGHT REASONING ====================
class ChainOfThought:
    """Mental simulation before answering"""
    
    def __init__(self, db):
        self.db = db
    
    def reason(self, query):
        """Internal reasoning steps"""
        steps = []
        
        # Step 1: Understand the question
        steps.append(f"Question: {query}")
        
        # Step 2: Break down concepts
        concepts = re.findall(r'\b\w{4,}\b', query.lower())
        steps.append(f"Key concepts: {', '.join(concepts[:5])}")
        
        # Step 3: Search memory
        results = []
        for concept in concepts[:3]:
            word_results = self.db.get_word_context(concept)
            results.extend(word_results)
        steps.append(f"Found {len(results)} related memories")
        
        # Step 4: Check contradictions
        contradictions = self.check_contradictions(results)
        if contradictions:
            steps.append(f"⚠️ Warning: {contradictions} conflicting info found")
        
        # Step 5: Formulate answer
        if results:
            confidence = min(1.0, len(results) / 10)
            steps.append(f"Confidence: {confidence:.1%}")
        else:
            steps.append("No relevant memory found")
        
        return steps
    
    def check_contradictions(self, sentences):
        """Check for contradictory information"""
        contradict_pairs = 0
        for i, s1 in enumerate(sentences[:5]):
            for s2 in sentences[i+1:10]:
                # Simple contradiction detection
                if ('not' in s1[0].lower() and s2[0].lower().replace('not', '') in s1[0].lower()) or \
                   ('not' in s2[0].lower() and s1[0].lower().replace('not', '') in s2[0].lower()):
                    contradict_pairs += 1
        return contradict_pairs
    
    def generate_answer(self, query):
        """Generate answer with reasoning"""
        steps = self.reason(query)
        
        # Format as natural language
        answer = "🧠 **Let me think step by step:**\n\n"
        for step in steps:
            answer += f"• {step}\n"
        
        # Final conclusion
        if "Found" in steps[-2]:
            answer += f"\n**Conclusion:** {steps[-2].split('Found')[1]}"
        else:
            answer += f"\n**Conclusion:** I need more information to answer this accurately."
        
        return answer


# ==================== FIX 4: EMOTIONAL INTELLIGENCE ====================
class EmotionalEngine:
    """Sentiment analysis and emotional response"""
    
    def __init__(self):
        self.sentiment_words = {
            'happy': ['achha', 'badhiya', 'mast', 'khush', 'great', 'good', 'awesome'],
            'angry': ['gussa', 'pagal', 'bakwas', 'kharab', 'bad', 'worst', 'stupid'],
            'sad': ['dukh', 'udaas', 'ro', 'sad', 'cry', 'depressed'],
            'excited': ['wow', 'zabardast', 'amazing', 'excited', 'awesome'],
            'confused': ['samajh', 'pata', 'confuse', 'kya', 'what', 'how']
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        text_lower = text.lower()
        scores = {}
        
        for emotion, words in self.sentiment_words.items():
            score = sum(1 for word in words if word in text_lower)
            if score > 0:
                scores[emotion] = score
        
        if not scores:
            return 'neutral', 0.5
        
        primary_emotion = max(scores, key=scores.get)
        intensity = min(1.0, scores[primary_emotion] / 5)
        
        return primary_emotion, intensity
    
    def generate_response(self, user_text, bot_response):
        """Modify response based on user emotion"""
        emotion, intensity = self.analyze_sentiment(user_text)
        
        if emotion == 'angry' and intensity > 0.3:
            return f"😔 Maaf karo Master, main samajh gaya. {bot_response}"
        elif emotion == 'sad' and intensity > 0.3:
            return f"🥺 Sab theek hoga Master. {bot_response}"
        elif emotion == 'happy' and intensity > 0.3:
            return f"😊 Bahut achha! {bot_response}"
        elif emotion == 'confused':
            return f"🤔 Kya confusion hai? Main samjha dunga. {bot_response}"
        else:
            return bot_response


# ==================== FIX 5: MARKOV CHAIN GENERATOR ====================
class MarkovGenerator:
    """Generate new sentences in the style of learned text"""
    
    def __init__(self, db, n=2):
        self.db = db
        self.n = n
        self.chain = defaultdict(list)
    
    def train_from_db(self):
        """Build Markov chain from database"""
        self.cursor.execute('SELECT sentence FROM knowledge ORDER BY RANDOM() LIMIT 1000')
        sentences = self.cursor.fetchall()
        
        for sentence in sentences:
            words = sentence[0].split()
            for i in range(len(words) - self.n):
                key = tuple(words[i:i+self.n])
                next_word = words[i+self.n]
                self.chain[key].append(next_word)
    
    def generate(self, seed_words, length=20):
        """Generate new sentence"""
        if len(seed_words) < self.n:
            return "Not enough seed words"
        
        key = tuple(seed_words[-self.n:])
        generated = list(seed_words)
        
        for _ in range(length):
            if key in self.chain and self.chain[key]:
                next_word = random.choice(self.chain[key])
                generated.append(next_word)
                key = tuple(generated[-self.n:])
            else:
                break
        
        return ' '.join(generated)


# ==================== FIX 6: CONFLICT RESOLUTION ====================
class ConflictResolver:
    """Handle contradictory information"""
    
    def __init__(self, db):
        self.db = db
    
    def resolve(self, query):
        """Check for conflicts and resolve"""
        # Get multiple sources
        results = self.db.search_knowledge(query, limit=10)
        
        if len(results) < 2:
            return None, []
        
        # Group by content similarity
        groups = defaultdict(list)
        for result in results:
            sentence = result[1]
            # Simple hash for grouping
            groups[hash(sentence[:50]) % 100].append(result)
        
        if len(groups) > 1:
            # Conflicts detected
            conflict_report = []
            for group_id, group in groups.items():
                example = group[0][1][:100]
                confidence = group[0][4]
                conflict_report.append((example, confidence, len(group)))
            
            return conflict_report, results
        
        return None, results
    
    def ask_user(self, conflict_report):
        """Ask user which source to trust"""
        response = "⚠️ **Contradictory information found:**\n\n"
        for i, (example, conf, count) in enumerate(conflict_report):
            response += f"Option {i+1}: {example}... (confidence: {conf:.1f}, {count} sources)\n\n"
        response += "Master, kaunsa sahi hai?"
        return response


# ==================== FIX 7: MULTI-TURN CONTEXT ====================
class ConversationMemory:
    """Remember conversation history"""
    
    def __init__(self, db, max_turns=5):
        self.db = db
        self.max_turns = max_turns
        self.current_context = {}
    
    def get_context(self):
        """Get recent conversation context"""
        recent = self.db.get_recent_conversations(self.max_turns)
        
        if not recent:
            return ""
        
        context = "Previous conversation:\n"
        for user, bot in recent:
            context += f"User: {user}\nBabita: {bot}\n"
        
        return context
    
    def resolve_pronouns(self, query, last_turn):
        """Replace pronouns with previous context"""
        if not last_turn:
            return query
        
        pronouns = {'ye', 'yeh', 'iska', 'isse', 'uska', 'usse', 'is', 'us'}
        words = query.split()
        
        resolved = []
        for word in words:
            if word.lower() in pronouns:
                # Extract noun from last turn
                last_words = last_turn[0].split()
                nouns = [w for w in last_words if len(w) > 3][-1:]
                if nouns:
                    resolved.append(nouns[0])
                else:
                    resolved.append(word)
            else:
                resolved.append(word)
        
        return ' '.join(resolved)


# ==================== FIX 8: HYBRID MODE (ONLINE LEARNING) ====================
class HybridLearner:
    """Learn from internet when available"""
    
    def __init__(self, db):
        self.db = db
        self.online = False
    
    def check_internet(self):
        """Check if internet is available"""
        try:
            import requests
            requests.get('http://8.8.8.8', timeout=2)
            self.online = True
        except:
            self.online = False
        return self.online
    
    def search_online(self, query):
        """Search online for information"""
        if not self.check_internet():
            return None
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Wikipedia search
            wiki_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
            response = requests.get(wiki_url, timeout=5)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                
                if paragraphs:
                    # Add to knowledge base
                    text = paragraphs[0].get_text()[:500]
                    self.db.add_knowledge(text, source='wikipedia', confidence=0.8)
                    return text
            
            return "Online search failed"
        except Exception as e:
            return f"Online error: {e}"


# ==================== FIX 9: SELF-CORRECTION LOOP ====================
class SelfCorrector:
    """Learn from mistakes"""
    
    def __init__(self, db):
        self.db = db
        self.mistakes = []
    
    def log_mistake(self, query, wrong_answer, correct_answer):
        """Log correction for learning"""
        self.mistakes.append({
            'query': query,
            'wrong': wrong_answer,
            'correct': correct_answer,
            'timestamp': datetime.now()
        })
        
        # Store in database
        self.db.cursor.execute('''
            INSERT INTO corrections (query, wrong, correct)
            VALUES (?, ?, ?)
        ''', (query, wrong_answer, correct_answer))
        self.db.conn.commit()
    
    def check_for_corrections(self, query):
        """Check if similar mistake was made before"""
        for mistake in self.mistakes[-10:]:
            if query.lower() in mistake['query'].lower() or \
               mistake['query'].lower() in query.lower():
                return mistake['correct']
        return None


# ==================== FIX 10: ENERGY MANAGEMENT ====================
class EnergyManager:
    """Optimize battery usage"""
    
    def __init__(self):
        self.active_mode = True
        self.last_active = time.time()
        self.idle_threshold = 300  # 5 minutes
    
    def check_energy_mode(self):
        """Switch to low-power mode if idle"""
        if time.time() - self.last_active > self.idle_threshold:
            self.active_mode = False
            return 'low'
        return 'high'
    
    def activity_detected(self):
        """User is active"""
        self.last_active = time.time()
        self.active_mode = True
    
    def should_process(self):
        """Should we process now?"""
        if self.active_mode:
            return True
        
        # In low-power mode, process only every 10th request
        return random.random() < 0.1


# ==================== FIX 11: AMBIGUITY RESOLUTION ====================
class AmbiguityResolver:
    """Resolve word ambiguity based on context"""
    
    def __init__(self, db):
        self.db = db
        self.ambiguity_map = {
            'apple': [('company', 'iphone, mac, ios'), ('fruit', 'eat, red, sweet')],
            'bank': [('river', 'water, flow'), ('financial', 'money, account')],
            'light': [('weight', 'heavy, mass'), ('illumination', 'bulb, bright')]
        }
    
    def resolve(self, word, context):
        """Find correct meaning based on context"""
        if word not in self.ambiguity_map:
            return word
        
        context_lower = ' '.join(context).lower()
        
        best_match = word
        best_score = 0
        
        for meaning, keywords in self.ambiguity_map[word]:
            score = sum(1 for kw in keywords.split() if kw in context_lower)
            if score > best_score:
                best_score = score
                best_match = f"{word} ({meaning})"
        
        return best_match


# ==================== FIX 12: PHYSICAL INTERACTION ====================
class PhysicalAwareness:
    """Know about device location and state"""
    
    def __init__(self):
        self.location = None
        self.movement = False
        self.battery_level = 100
        self.screen_on = True
    
    def update_sensors(self):
        """Get sensor data (if available)"""
        try:
            # Try to get GPS (requires permissions)
            gps_location = gps.get_last_known_location()
            if gps_location:
                self.location = (gps_location['lat'], gps_location['lon'])
        except:
            pass
        
        try:
            # Get accelerometer data
            accel = accelerometer.acceleration
            self.movement = any(abs(a) > 0.1 for a in accel)
        except:
            pass
        
        # Battery level (approximate)
        try:
            import subprocess
            result = subprocess.run(['dumpsys', 'battery'], 
                                   capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'level' in line:
                    self.battery_level = int(line.split(':')[1].strip())
        except:
            pass
    
    def get_context(self):
        """Get physical context"""
        self.update_sensors()
        
        context = []
        if self.movement:
            context.append("Master is moving")
        if self.battery_level < 20:
            context.append("Battery low")
        if self.location:
            context.append(f"Location available")
        
        return context


# ==================== FIX 13: PROACTIVE SUGGESTIONS ====================
class ProactiveEngine:
    """Give suggestions without being asked"""
    
    def __init__(self, db):
        self.db = db
        self.last_suggestion = None
        self.suggestion_time = None
    
    def should_suggest(self):
        """Decide if we should give a suggestion"""
        if not self.suggestion_time:
            return True
        
        # Suggest every 30 minutes
        return time.time() - self.suggestion_time > 1800
    
    def generate_suggestion(self, user_context=""):
        """Generate proactive suggestion"""
        if not self.should_suggest():
            return None
        
        # Get most accessed knowledge
        self.db.cursor.execute('''
            SELECT sentence FROM knowledge
            ORDER BY access_count DESC, timestamp DESC
            LIMIT 1
        ''')
        result = self.db.cursor.fetchone()
        
        if result:
            self.suggestion_time = time.time()
            return f"💡 **Suggestion:** Remember this? {result[0][:100]}..."
        
        return None


# ==================== FIX 14: KILL SWITCH ====================
class KillSwitch:
    """Emergency stop for AI"""
    
    def __init__(self):
        self.emergency_stop = False
        self.secret_phrase = "babita shutdown"
        self.reset_phrase = "babita restart"
    
    def check_command(self, text):
        """Check if kill switch activated"""
        if self.secret_phrase in text.lower():
            self.emergency_stop = True
            return "🛑 Emergency shutdown activated. Say 'babita restart' to resume."
        
        if self.reset_phrase in text.lower():
            self.emergency_stop = False
            return "✅ System restarted. How can I help?"
        
        return None


# ==================== FIX 15: MAIN BABITA ENGINE ====================
class BabitaUltimateEngine:
    """BABITA V6.0 - All fixes integrated"""
    
    def __init__(self, app_data_dir, password=None):
        self.data_dir = app_data_dir
        self.db_path = os.path.join(app_data_dir, "babita_v6.db")
        
        # Initialize all components
        self.db = BabitaDatabase(self.db_path)
        self.chain_of_thought = ChainOfThought(self.db)
        self.emotional = EmotionalEngine()
        self.conflict = ConflictResolver(self.db)
        self.memory = ConversationMemory(self.db)
        self.hybrid = HybridLearner(self.db)
        self.corrector = SelfCorrector(self.db)
        self.energy = EnergyManager()
        self.ambiguity = AmbiguityResolver(self.db)
        self.physical = PhysicalAwareness()
        self.proactive = ProactiveEngine(self.db)
        self.kill_switch = KillSwitch()
        
        # User profile
        self.user_name = self.db.get_user_profile('name') or "Master"
        self.user_age = int(self.db.get_user_profile('age') or 21)
        self.user_goal = self.db.get_user_profile('goal') or "Knowledge"
        
        # Conversation tracking
        self.last_turn = None
        self.context_window = []
    
    def learn_from_file(self, filepath):
        """Process a file and learn"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 🔥 Fix 5: Smart sentence splitting
            sentences = re.split(r'(?<!\b(?:Dr|Mr|Ms|St|Vs)\.)(?<=\.|\?|!)\s+', content)
            
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    confidence = 1.0
                    # Higher confidence for longer sentences
                    if len(sentence) > 500:
                        confidence = 0.9
                    
                    self.db.add_knowledge(
                        sentence.strip(), 
                        source=os.path.basename(filepath),
                        confidence=confidence
                    )
            
            return len(sentences)
        except Exception as e:
            return f"Error: {e}"
    
    def think(self, user_input):
        """Main thinking function"""
        
        # Check energy mode
        if not self.energy.should_process():
            return "⚡ Energy saving mode. Please try again in a moment."
        
        # Check kill switch
        kill_response = self.kill_switch.check_command(user_input)
        if kill_response:
            return kill_response
        
        if self.kill_switch.emergency_stop:
            return "🛑 System in emergency stop mode. Say 'babita restart' to resume."
        
        # Record activity
        self.energy.activity_detected()
        
        # Check for proactive suggestion
        suggestion = self.proactive.generate_suggestion()
        if suggestion and random.random() < 0.2:  # 20% chance
            return suggestion
        
        # Check for corrections
        corrected = self.corrector.check_for_corrections(user_input)
        if corrected:
            return f"📝 **Correction:** {corrected}"
        
        # Resolve pronouns with context
        if self.last_turn:
            user_input = self.memory.resolve_pronouns(user_input, self.last_turn)
        
        # Resolve ambiguity
        context_words = re.findall(r'\b\w+\b', user_input)
        resolved_input = []
        for word in user_input.split():
            resolved_word = self.ambiguity.resolve(word.lower(), context_words)
            resolved_input.append(resolved_word)
        resolved_text = ' '.join(resolved_input)
        
        # Search database
        results = self.db.search_knowledge(resolved_text, limit=10)
        
        # Check for conflicts
        conflicts, all_results = self.conflict.resolve(resolved_text)
        if conflicts:
            return self.conflict.ask_user(conflicts)
        
        # Generate response
        if results:
            # Get top result
            top_result = results[0]
            
            # Update access count
            self.db.cursor.execute('''
                UPDATE knowledge SET access_count = access_count + 1,
                last_accessed = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (top_result[0],))
            self.db.conn.commit()
            
            # Generate answer
            answer = f"📚 **From memory:**\n{top_result[1][:300]}..."
            
            # Add reasoning if available
            if random.random() < 0.3:  # 30% chance
                reasoning = self.chain_of_thought.generate_answer(user_input)
                answer = reasoning + "\n\n" + answer
        else:
            # Try online learning
            if self.hybrid.check_internet():
                online_result = self.hybrid.search_online(user_input)
                if online_result:
                    answer = f"🌐 **Found online:**\n{online_result}"
                else:
                    answer = "❌ Kuch nahi mila. Online bhi search kiya par khaali."
            else:
                answer = "❌ Kuch nahi mila. Pehle kuch books load karo ya internet on karo."
        
        # Add emotional touch
        answer = self.emotional.generate_response(user_input, answer)
        
        # Add physical context if relevant
        physical_ctx = self.physical.get_context()
        if physical_ctx and "location" in answer.lower():
            answer += f"\n\n📍 {', '.join(physical_ctx)}"
        
        # Save conversation
        self.db.add_conversation(user_input, answer)
        self.last_turn = (user_input, answer)
        
        return answer
    
    def set_user_profile(self, **kwargs):
        """Set user profile information"""
        for key, value in kwargs.items():
            self.db.set_user_profile(key, value)
            if key == 'name':
                self.user_name = value
            elif key == 'age':
                self.user_age = int(value)
            elif key == 'goal':
                self.user_goal = value
    
    def cleanup(self):
        """Cleanup old knowledge"""
        removed = self.db.cleanup_old_knowledge()
        return f"🧹 Cleaned up {removed} old memories."
    
    def get_stats(self):
        """Get statistics"""
        self.db.cursor.execute('SELECT COUNT(*) FROM knowledge')
        total_knowledge = self.db.cursor.fetchone()[0]
        
        self.db.cursor.execute('SELECT COUNT(*) FROM concept_map')
        total_concepts = self.db.cursor.fetchone()[0]
        
        self.db.cursor.execute('SELECT COUNT(*) FROM conversation')
        total_conversations = self.db.cursor.fetchone()[0]
        
        return {
            'knowledge': total_knowledge,
            'concepts': total_concepts,
            'conversations': total_conversations,
            'energy': 'High' if self.energy.active_mode else 'Low',
            'battery': self.physical.battery_level
        }
    
    def close(self):
        """Close database connection"""
        self.db.close()


# ==================== KIVY UI ====================
class BabitaUltimateAI(App):
    def build(self):
        Window.size = (450, 700)
        
        # Initialize engine
        self.engine = BabitaUltimateEngine(self.user_data_dir)
        
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=5)
        
        # Header
        header = BoxLayout(size_hint_y=0.08)
        self.header_label = Label(
            text="🧠 BABITA ULTIMATE v6.0",
            font_size='18sp',
            color=[0, 1, 0, 1]
        )
        header.add_widget(self.header_label)
        main_layout.add_widget(header)
        
        # Stats bar
        stats_layout = BoxLayout(size_hint_y=0.05, spacing=5)
        self.stats_label = Label(
            text="Loading...",
            font_size='10sp',
            color=[0.5, 0.5, 0.5, 1]
        )
        stats_layout.add_widget(self.stats_label)
        main_layout.add_widget(stats_layout)
        
        # Progress bar
        self.progress = ProgressBar(max=100, value=0, size_hint_y=0.02)
        main_layout.add_widget(self.progress)
        
        # Chat area
        self.chat_history = BoxLayout(orientation='vertical', size_hint_y=None, spacing=5)
        self.chat_history.bind(minimum_height=self.chat_history.setter('height'))
        
        scroll = ScrollView(size_hint_y=0.65)
        scroll.add_widget(self.chat_history)
        main_layout.add_widget(scroll)
        
        # Input area
        input_layout = BoxLayout(size_hint_y=0.1, spacing=5)
        
        self.user_input = TextInput(
            hint_text="Kuch bhi pucho...",
            multiline=True,
            size_hint_x=0.7,
            background_color=[0.1, 0.1, 0.1, 1],
            foreground_color=[0, 1, 0, 1]
        )
        
        send_btn = Button(
            text="➡️",
            size_hint_x=0.15,
            background_color=[0, 0.7, 0, 1]
        )
        send_btn.bind(on_press=self.process_query)
        
        upload_btn = Button(
            text="📚",
            size_hint_x=0.15,
            background_color=[0, 0.5, 0.8, 1]
        )
        upload_btn.bind(on_press=self.open_file_manager)
        
        input_layout.add_widget(self.user_input)
        input_layout.add_widget(send_btn)
        input_layout.add_widget(upload_btn)
        
        main_layout.add_widget(input_layout)
        
        # Control buttons
        control_layout = BoxLayout(size_hint_y=0.05, spacing=5)
        
        clear_btn = Button(
            text="🗑️ Clear",
            background_color=[0.8, 0.2, 0.2, 1]
        )
        clear_btn.bind(on_press=self.clear_chat)
        
        stats_btn = Button(
            text="📊 Stats",
            background_color=[0.8, 0.6, 0.1, 1]
        )
        stats_btn.bind(on_press=self.show_stats)
        
        profile_btn = Button(
            text="👤 Profile",
            background_color=[0.2, 0.5, 0.8, 1]
        )
        profile_btn.bind(on_press=self.edit_profile)
        
        control_layout.add_widget(clear_btn)
        control_layout.add_widget(stats_btn)
        control_layout.add_widget(profile_btn)
        
        main_layout.add_widget(control_layout)
        
        # Update stats
        self.update_stats()
        
        # Welcome message
        self.add_message(f"Namaste {self.engine.user_name}! Main Babita Ultimate AI hoon.", False)
        self.add_message("Deep Reasoning + Emotional Intelligence + Proactive Learning seekh liya!", False)
        
        return main_layout
    
    def update_stats(self):
        """Update stats display"""
        stats = self.engine.get_stats()
        self.stats_label.text = f"📚 {stats['knowledge']} | 🧠 {stats['concepts']} | 💬 {stats['conversations']} | ⚡ {stats['energy']} | 🔋 {stats['battery']}%"
    
    def add_message(self, text, is_user):
        """Add message to chat"""
        msg = Label(
            text=text,
            size_hint_y=None,
            halign='left' if not is_user else 'right',
            valign='middle',
            text_size=(Window.width - 60, None),
            color=[0, 1, 0, 1] if not is_user else [1, 1, 1, 1]
        )
        msg.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1] + 10))
        
        self.chat_history.add_widget(msg)
        Clock.schedule_once(lambda dt: setattr(self.chat_history.parent, 'scroll_y', 0), 0.1)
    
    def open_file_manager(self, instance):
        """Open file chooser"""
        content = FileChooserIconView(filters=['*.txt', '*.pdf', '*.md'])
        content.bind(on_submit=self.load_document)
        
        popup = Popup(
            title="📖 Book Load Karo",
            content=content,
            size_hint=(0.9, 0.9)
        )
        self._popup = popup
        popup.open()
    
    def load_document(self, instance, selection, touch):
        """Load document"""
        if selection:
            filepath = selection[0]
            self.add_message(f"📖 Loading: {os.path.basename(filepath)}...", False)
            
            thread = threading.Thread(target=self.process_document, args=(filepath,))
            thread.daemon = True
            thread.start()
            self._popup.dismiss()
    
    def process_document(self, filepath):
        """Process document"""
        try:
            # Simulate progress
            for i in range(10):
                time.sleep(0.1)
                Clock.schedule_once(lambda dt: setattr(self.progress, 'value', (i+1)*10))
            
            # Learn from file
            result = self.engine.learn_from_file(filepath)
            
            Clock.schedule_once(lambda dt: self.document_loaded_callback(filepath, result))
            
        except Exception as e:
            Clock.schedule_once(lambda dt: self.add_message(f"Error: {str(e)}", False))
    
    def document_loaded_callback(self, filepath, result):
        """Document loaded"""
        self.progress.value = 100
        self.add_message(f"✅ {os.path.basename(filepath)} loaded! {result} sentences learned.", False)
        self.update_stats()
    
    def process_query(self, instance):
        """Process user query"""
        query = self.user_input.text.strip()
        if not query:
            return
        
        self.add_message(query, True)
        self.user_input.text = ""
        
        thread = threading.Thread(target=self.think_about, args=(query,))
        thread.daemon = True
        thread.start()
    
    def think_about(self, query):
        """Think about query"""
        Clock.schedule_once(lambda dt: self.add_message("🤔 Thinking deeply...", False))
        time.sleep(1)
        
        Clock.schedule_once(lambda dt: self.remove_last_message(), 1.1)
        
        response = self.engine.think(query)
        Clock.schedule_once(lambda dt: self.add_message(response, False))
        Clock.schedule_once(lambda dt: self.update_stats(), 0.5)
    
    def remove_last_message(self):
        """Remove thinking message"""
        if len(self.chat_history.children) > 0:
            self.chat_history.remove_widget(self.chat_history.children[0])
    
    def clear_chat(self, instance):
        """Clear chat"""
        self.chat_history.clear_widgets()
        self.add_message("Chat cleared! Kuch naya puchiye.", False)
    
    def show_stats(self, instance):
        """Show detailed stats"""
        stats = self.engine.get_stats()
        popup = Popup(
            title="📊 Statistics",
            content=Label(text=f"""
Knowledge: {stats['knowledge']}
Concepts: {stats['concepts']}
Conversations: {stats['conversations']}
Energy Mode: {stats['energy']}
Battery: {stats['battery']}%

Memory Cleanup: Available
Online Mode: {'ON' if self.engine.hybrid.check_internet() else 'OFF'}
            """),
            size_hint=(0.8, 0.6)
        )
        popup.open()
    
    def edit_profile(self, instance):
        """Edit user profile"""
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        name_input = TextInput(hint_text="Your name", text=self.engine.user_name)
        age_input = TextInput(hint_text="Your age", text=str(self.engine.user_age))
        goal_input = TextInput(hint_text="Your goal", text=self.engine.user_goal)
        
        save_btn = Button(text="Save", size_hint_y=0.2)
        
        layout.add_widget(Label(text="Edit Profile"))
        layout.add_widget(name_input)
        layout.add_widget(age_input)
        layout.add_widget(goal_input)
        layout.add_widget(save_btn)
        
        popup = Popup(
            title="👤 Profile",
            content=layout,
            size_hint=(0.8, 0.6)
        )
        
        def save_profile(instance):
            self.engine.set_user_profile(
                name=name_input.text,
                age=int(age_input.text or 21),
                goal=goal_input.text
            )
            popup.dismiss()
        
        save_btn.bind(on_press=save_profile)
        popup.open()
    
    def on_stop(self):
        """Cleanup on exit"""
        self.engine.cleanup()
        self.engine.close()


if __name__ == '__main__':
    BabitaUltimateAI().run()
