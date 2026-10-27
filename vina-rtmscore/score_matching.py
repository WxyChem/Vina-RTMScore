import math


def match_scores(expected, actual, values):
    if len(actual)!=len(values) or len(set(actual))!=len(actual) or set(expected)!=set(actual):
        raise ValueError('Scored IDs do not exactly match candidate IDs')
    result = dict(zip(actual,map(float,values)))
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Non-finite RTMScore')
    return [result[k] for k in expected]
