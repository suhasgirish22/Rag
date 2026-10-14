# Notes

One file per lesson. Four lines each, two minutes of typing.

This folder is the completion record for the course. The tick box on the site used
to live in browser local storage, which meant nobody could see it but the browser
that stored it. A note committed here is visible from anywhere, so it is what the
site counts and what tells me whether a lesson was pitched right.

## How to fill one in

Open the lesson on
[the curriculum page](https://suhasgirish22.github.io/Rag/curriculum.html) and use
the "Write your note" button, or go straight to the file:

    https://github.com/suhasgirish22/Rag/edit/gh-pages/notes/day-031.md

GitHub's editor works fine on a phone. Type, scroll down, hit "Commit changes".

Three things happen:

1. The day turns solid on the progress grid, because the file now has words in it.
2. The commit lands on your contribution graph, dated today.
3. Your clarity rating reaches me, and lessons that score low get rewritten.

## The clarity rating

`**Clarity, 1 to 5:**` followed by a digit. 1 means it lost you, 5 means it was
easy. This is the only control you have over how hard the course gets, so use it
honestly. Anything rated 1 or 2 gets a plainer rewrite and a republish.

## How the count is produced

`tools/sync_progress.py` walks this folder, treats a file as answered if it has
words in it beyond the blank template, and writes `progress.json` at the repo
root. The page reads that file. Nothing about it is clever and you can check it
yourself.
