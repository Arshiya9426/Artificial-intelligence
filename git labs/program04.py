class RuleBasedSystem:
    def __init__(self, facts, rules):
        self.facts = set(facts)
        self.rules = rules

    def forward_chain(self):
        iterations = 0

        while True:
            new_fact_added = False

            for rule in self.rules:
                # Check if all conditions of the rule exist in current facts
                if all(condition in self.facts for condition in rule['if']):
                    if rule['then'] not in self.facts:
                        print(
                            f"Rule Triggered: IF {rule['if']} "
                            f"THEN Add {rule['then']}"
                        )

                        self.facts.add(rule['then'])
                        new_fact_added = True

            if not new_fact_added:
                break

            iterations += 1

        print("\nNumber of iterations:", iterations)
        return self.facts


# Initial facts
facts = [
    "It is raining",
    "The ground is wet"
]

# Rules
rules = [
    {
        'if': ["It is raining"],
        'then': "Take an umbrella"
    },
    {
        'if': ["The ground is wet"],
        'then': "Drive carefully"
    },
    {
        'if': ["It is raining", "The ground is wet"],
        'then': "Weather is bad"
    }
]

# Create the rule-based system
system = RuleBasedSystem(facts, rules)

# Apply forward chaining
result = system.forward_chain()

# Display final facts
print("\nFinal Facts:")
for fact in result:
    print("-", fact)