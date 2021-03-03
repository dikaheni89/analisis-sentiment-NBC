# Facebook crawling with Python

> Demo: https://www.youtube.com/watch?v=Fx0UWOzYsig

## Features:

-   Get information of posts
-   Filter comments
-   Not required sign in

## Data Fields:

```json
[
    {
        "url": "",
        "id": "",
        "utime": "",
        "text": "",
        "total_shares": "",
        "total_cmts": "",
        "reactions": ["reactions displayed below post content"],
        "crawled_cmts": [
            {
                "id": "",
                "utime": "",
                "user_url": "",
                "user_id": "",
                "user_name": "",
                "text": "",
                "replies": [
                    { "id": "", "utime": "", "user_id": "", "user_name": "",  "text": "" },
                    { "id": "", "utime": "", "user_id": "", "user_name": "",  "text": "" },
                    { "id": "", "utime": "", "user_id": "", "user_name": "",  "text": "" },
                ]
            },
        ]
    },
]
```
        
## Usage:

1. Install Helium: `pip install helium`
2. Customize the `crawler.py` file:
    - **PAGE_URL**: url of Facebook page
    - **SCROLL_DOWN**: number of scroll times for loading more posts 
    - **FILTER_CMTS_BY**: show comments by `MOST_RELEVANT` / `NEWEST` / `ALL_COMMENTS`
    - **VIEW_MORE_CMTS**: number of times for loading more comments
    - **VIEW_MORE_REPLIES**: number of times for loading more replies
3. Start crawling: 
    - Sign out Facebook (cause some CSS Selectors will be different as sign in)
    - Run `python crawler.py`

**Reference:** https://github.com/mherrmann/selenium-python-helium
