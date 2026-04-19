import sys

NL = chr(10)
p = '/home/josh/The-Inference-Difference/inference_difference/model_client.py'
ls = open(p).readlines()

# 1. Changelog
for i, l in enumerate(ls):
    if '# ---- Changelog ----' in l:
        if '#173' not in ls[i+1]:
            ls.insert(i+1, '# [2026-04-19] CC Sonnet 4.6 -- #173 gap (b): 429 rate-limit caching' + NL)
        break

# 2. Rate-limit dicts in __init__ (after _load_policy_blocked_cache call)
for i, l in enumerate(ls):
    if 'self._load_policy_blocked_cache()' in l:
        if '_rate_limited' not in ls[i+1]:
            ls.insert(i+1, '        self._rate_limited = {}   # model_name -> unblock_timestamp (#173b)' + NL)
            ls.insert(i+2, '        self._rate_limit_hits = {}  # consecutive 429 count per model' + NL)
        break

# 3. Add is_rate_limited methods before call()
for i, l in enumerate(ls):
    if '    def call(' in l:
        if 'is_rate_limited' not in ls[i-2]:
            ins = [
                '    def is_rate_limited(self, model_name: str) -> bool:' + NL,
                '        unblock = self._rate_limited.get(model_name)' + NL,
                '        if unblock is None: return False' + NL,
                '        if time.time() < unblock: return True' + NL,
                '        del self._rate_limited[model_name]' + NL,
                '        self._rate_limit_hits.pop(model_name, None)' + NL,
                '        return False' + NL,
                '' + NL,
                '    def is_rate_limited_by_id(self, model_id: str) -> bool:' + NL,
                '        _, _, mn = _resolve_provider(model_id)' + NL,
                '        return self.is_rate_limited(mn)' + NL,
                '' + NL,
            ]
            for j, item in enumerate(ins):
                ls.insert(i+j, item)
        break

# 4. Change if->elif on 404/403 block and insert 429 block before it
for i, l in enumerate(ls):
    if 'if e.code in (404, 403)' in l and 'data policy' in l:
        ls[i] = l.replace('            if e.code in (404', '            elif e.code in (404')
        if 'e.code == 429' not in ls[i-1]:
            ins = [
                '            if e.code == 429:' + NL,
                '                hits = self._rate_limit_hits.get(model_name, 0) + 1' + NL,
                '                self._rate_limit_hits[model_name] = hits' + NL,
                '                backoff_s = min(60 * (2 ** (hits - 1)), 1800)' + NL,
                '                self._rate_limited[model_name] = time.time() + backoff_s' + NL,
                '                logger.info(' + NL,
                '                    "Model %s 429 hit #%d -- cooldown %.0fs",' + NL,
                '                    model_name, hits, backoff_s,' + NL,
                '                )' + NL,
            ]
            for j, item in enumerate(ins):
                ls.insert(i+j, item)
        break

open(p, 'w').writelines(ls)
print('MCLIENT_PATCHED')
