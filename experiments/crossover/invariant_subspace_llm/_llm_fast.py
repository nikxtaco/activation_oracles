"""Make ao_analyzer's Gemini calls fast: disable Gemini-3's default extended 'thinking'.

gemini-3-flash-preview defaults to extended reasoning, so each call takes ~60-80s. The investigator
and judge are simple classification tasks that don't need it; passing reasoning_effort='none' drops
a call to ~1s. We patch llm.get_client (which llm.chat calls on every request, so this covers the
agent and judge too) to return a client that defaults reasoning_effort and has a sane timeout.
"""


def patch_fast(llm, timeout=90.0, effort="none", rpm=None):
    if rpm is not None:                       # raise the client-side rate cap to use many workers
        llm.PROVIDERS["google"]["rpm"] = rpm
    orig_get_client = llm.get_client

    def get_client():
        c = orig_get_client().with_options(timeout=timeout, max_retries=0)
        orig_create = c.chat.completions.create

        def create(*args, **kwargs):
            kwargs.setdefault("reasoning_effort", effort)
            return orig_create(*args, **kwargs)

        c.chat.completions.create = create
        return c

    llm.get_client = get_client
