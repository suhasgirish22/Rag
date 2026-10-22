# Notes

One file per lesson, holding your reply to that lesson's email.

You do not need to edit anything here. When you finish a lesson, reply to its
email with one line: "done", or the first word or sentence that confused you.
The next morning your reply is copied into `day-NNN.md` and committed under your
GitHub account, and only then is the next lesson sent.

So every square on your contribution graph from this folder is a day you
actually did a lesson. Days with no reply produce no commit.

`tools/sync_progress.py` reads this folder and writes `progress.json`, which is
what the curriculum page uses for the count and the streak.

The first version of the course, 39 lessons that were pitched far too high, is
kept in `archive/v1/` in case it is ever useful as reference. It is not part of
the course any more.
