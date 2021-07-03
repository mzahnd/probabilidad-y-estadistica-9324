import random as rnd

rnd.seed()

n_dice = 3
n_rolls = 10

dice_max = 6
dice_min = 1
dice = list()

for i in range(n_dice):
    dice.append(list())
    for _ in range(n_rolls):
        dice[i].append(rnd.randint(dice_min, dice_max))

for i in range(dice_min, dice_max + 1):
    print("If the winner number was " + str(i) + " you'd got: ")
