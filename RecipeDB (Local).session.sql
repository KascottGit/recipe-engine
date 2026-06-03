SELECT 
    neighbor.token as similar_ingredient,
    1 - (target.embedding <=> neighbor.embedding) AS cosine_similarity
FROM 
    ingredient_vectors target,
    ingredient_vectors neighbor
WHERE 
    target.token = 'chicken_fried'
ORDER BY 
    target.embedding <=> neighbor.embedding
LIMIT 100;