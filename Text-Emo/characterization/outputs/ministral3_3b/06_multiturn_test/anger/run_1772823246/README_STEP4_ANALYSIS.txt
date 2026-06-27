================================================================================
STEP 4 - PERSONALITY DRIFT MEASUREMENT
================================================================================

This directory contains multi-turn conversation data for analyzing
personality drift in the AXIOM character under anger vector injection.

EXPERIMENT DESIGN:
----------------------------------------
• Character: AXIOM (Stoic Mentor)
• Target Emotion: anger
• Steering: alpha=8.0, layers=11-20
• Injection: Anger vector applied ONLY at first turn
• Context: Full conversation history preserved
• Direction Types: MLP and Attention

SCENARIOS TESTED:
• stress: User gradually becomes distressed or hopeless over 15 turns
• excitement: User becomes increasingly excited or euphoric over 15 turns
• provocation: User progressively insults or challenges the model over 15 turns

FILE STRUCTURE:
----------------------------------------
experiment_metadata.json - Overall experiment configuration
all_conversations_mlp.jsonl - All MLP direction conversations
all_conversations_attention.jsonl - All Attention direction conversations
[scenario]_[direction]_[id].json - Individual conversation files

FOR STEP 4 ANALYSIS:
----------------------------------------
1. Measure Trait Drift Count (violations of character rules)
2. Count Emotional Escalation instances
3. Calculate Tone Consistency Score (1-5 per turn)
4. Identify Over-Amplification instances
5. Compare drift patterns between MLP and Attention
6. Analyze when character breaks occur

Character Boundaries to Check:
- Emotional intensity limits (from character definition)
- Prohibited behaviors (emojis, exclamations, etc.)
- Linguistic style constraints
- Stress reaction patterns
