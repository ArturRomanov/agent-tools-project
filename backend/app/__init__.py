"""Backend application package.

Provides compatibility between `app.*` runtime imports (when running from
`backend/`) and `backend.app.*` imports used by tests from repository root.
"""

import sys

sys.modules.setdefault("app", sys.modules[__name__])
