SELECT 
    target.token as target_ingredient,
    neighbor.token as similar_ingredient,
    1 - (target.embedding <=> neighbor.embedding) AS cosine_similarity
FROM 
    ingredient_vectors target,
    ingredient_vectors neighbor
WHERE 
    target.token = 'chili_powder'
    AND neighbor.token != 'chili_powder'
ORDER BY 
    target.embedding <=> neighbor.embedding
LIMIT 100;