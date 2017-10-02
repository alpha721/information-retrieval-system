from vector_space import VectorSpace

vector_space = VectorSpace(["The cat in the hat disabled", "A cat is a fine pet ponies.", "Dogs and cats make good pets.","I haven't got a hat."])

#Search for cat
print vector_space.search(["cat"])

#Show score for relatedness against document 0
print vector_space.related(0) 
