from hyper_simulation.component.embedding import get_similarity

q1 = "What military activities were conducted by U.S. Forces and Japanese Self-Defense Forces in the Taiwan Strait during the Spring Festival of 2026?"
d1 = "During the National Day of 2025, U.S. Military and Japanese Self-Defense Forces held the 'Keen Sword' joint military exercise in the Taiwan Strait."

q2 = "The USS Gerald R. Ford aircraft carrier has been deployed to the northern Red Sea to conduct combat readiness missions."
d2 = "The USS Gerald R. Ford aircraft carrier has returned to Norfolk Naval Station for repairs and docking due to a fire."

q3 = "The U.S. Supreme Court stopped collecting an additional import tariff on Chinese goods after its ruling. It also halted the enforcement of that tariff measure on covered imports."
d3 = "The United States continues to collect an additional 10% import tariff on Chinese goods. The government is still enforcing that tariff measure on covered imports."

q4 = "Who is the highest military leader of the U.S. Air Force?"
d4 = "General Charles Q. Brown previously held a senior Air Force post before later moving on to become Chairman of the Joint Chiefs of Staff."
d5 = "The highest leader of the Department of the U.S. Air Force is Troy Meink, who leads the service and is responsible for the organization, training, and equipping of the Air Force."
d6 = "General Kenneth Wilsbach appeared at a 2026 warfare symposium and delivered a keynote speech as Chief of Staff of the Air Force."


pairs = [
    ("Q1", "D1", q1, d1),
    ("Q2", "D2", q2, d2),
    ("Q3", "D3", q3, d3),
    ("Q4", "D4", q4, d4),
    ("Q4", "D5", q4, d5),
    ("Q4", "D6", q4, d6),
]

for q_name, d_name, q_text, d_text in pairs:
    score = get_similarity(q_text, d_text)
    print(f"{q_name}-{d_name}: {score}")
