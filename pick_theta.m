function para = pick_theta(lmd,b,bw)

para = [];

for i = 1: length(lmd),
    for j = 1: length(b),
        if abs(lmd(i)*b(j)) <= bw,
            para = [para; lmd(i), b(j);];
        end
    end
end

para = para';