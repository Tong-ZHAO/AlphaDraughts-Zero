class Elo():
	def __init__(self):
		pass

	@staticmethod
	def expected_score(rating_player, rating_reference, c=400):
		'''
		returns the expected score of the player when it plays 
		against a certain reference. c is a constant parameter.
		'''
		return 1 / (1 + 10 ** ((rating_reference - rating_player) / c))

	@staticmethod
	def actual_score(record, draw_point=0.5):
		'''
		returns the actual score of the player.

		From the perspective of the player,
		record[0]: number of wins
		record[1]: number of loses
		record[2]: number of draws
		'''
		return (record[0] + draw_point * record[2]) / sum(record)

	@staticmethod
	def new_score(rating_player, actual_score_, expected_score_, K=32):
		'''
		returns the updated score of the player if he actually gets
		actual_score_ but his expected score is expected_score_. 
		K is a constant parameter.
		'''
		return rating_player + K * (actual_score_ - expected_score_)

if __name__ == "__main__":
	print("An example:")
	score_player = 1456
	score_reference = 1503
	record = [11, 13, 6] # 11 wins, 13 loses, 6 draws
	print("Before these games, player's elo rating is", score_player)

	expected = Elo.expected_score(score_player, score_reference)
	actual = Elo.actual_score(record)
	score_player = Elo.new_score(score_player, actual, expected)
	print("After these games, player's elo rating is", score_player)