

def read_ann(ann):
    """Read an annotation file.

    Input:
    ann -- a filename

    Output:
    entities -- a list of tuples (id, type, ranges) where type is the entity type and
    ranges is a list of pairs (i, j),
    """
    with open(ann) as f:
        entities = []
        for line in f:
            if line[0] == 'T':
                sline = line.split()
                ranges = []
                n = 3
                i = sline[2]
                j = sline[3]
                while ';' in j:
                    j, i2 = sline[n].split(';')
                    ranges.append((int(i), int(j)))
                    i = i2
                    j = sline[n+1]
                    n += 1
                ranges.append((int(i), int(j)))

                # sort ranges
                sorted_ranges = sorted(ranges, key=lambda r: r[0])
                e = (sline[0], sline[1], sorted_ranges)
                entities.append(e)

        sorted_entities = sorted(entities, key=lambda e: e[1][0][0])
        return sorted_entities
