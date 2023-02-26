from NumPyCreator import NumPyCreator

npc = NumPyCreator()
npc.from_list([[1,2,3],[6,3,4]])
npc.from_tuple(("a", "b", "c"))
npc.from_iterable(range(5))
shape=(3,5)
npc.from_shape(shape)
npc.random(shape)
npc.identity(4)