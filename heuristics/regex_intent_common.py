import re

WH = r"(what|when|where|who|whom|which|whose|why|how)"
MODAL = r"(must|shall|will|should|would|can|could|may|might)"
MAX_1_WORDS = r" *((\w|')+\b *){0,1}"
MAX_2_WORDS = r" *((\w|')+\b *){0,2}"
MAX_3_WORDS = r" *((\w|')+\b *){0,3}"
MAX_4_WORDS = r" *((\w|')+\b *){0,4}"
MAX_6_WORDS = fr"{MAX_3_WORDS}{MAX_3_WORDS}"
MIN_2_WORDS = r" *((\w|')+\b *){2,}"
PPL = r"(actor|actress|author|writer|player|team)(s|es)?"
THINGS = r"(game|movie|book|place|show|song|name)(s|es)?"
QUESTION_START = fr"(((have|did) you|there|(i|i'm)) {MAX_2_WORDS} ((of|about) )?)"
DURATION = fr"(for )?(like )?(just )?(about )?(a )?{MAX_1_WORDS}(sec( |$)|seconds?(?! time)|minutes?|moment)"
YOU_DO = r"(^|(you|we|please|but|so|all right|wait|yeah|yes) +)"
USER_DO = fr"(can i have|give me|think( about( \w+)?)?|back{MAX_2_WORDS}in|need to go|come back)"
START = r"((so|(no\b ?){1,3}|yeah|sure|yes|okay|because|well|but|please|wait|huh|shush|hey|wow|really|oh( my goodness)?) *)?"
SORRY = r"( ?(i'm sorry( to interrupt)?|sorry|my apologies) *)"
MY_THINGS = fr"my{MAX_2_WORDS}({PPL}|{THINGS})( to{MAX_1_WORDS})?"
FRONT_MOD = fr"( ?(a|an|the|this) {MAX_1_WORDS})"
DEVICE = fr"( ?\b(amazon|alexa|echo|alexus|all|you|social bot)\b ?)"
GOING_TO_BED = fr"{MAX_1_WORDS}((gonna have|need|have|'ve|like|'m (about|ready|getting ready|trying)) to )+go to bed{MAX_3_WORDS}"

CONTINUE = r"go for it|continue|go ahead|go for it|keep going|let's hear it|read it all|if you.*(like|want)"

IT_IS = r"(this|that|it)('s| is| was| sounds)"
YOU_ARE = r"you('re| are)"
EXCLUDE_NOT = r"(?!.*not )"
EXCLUDE_NOT_AFTER = r"(?!.*not)"
DO_NOT = r"(do|did)(n't| not)"
HAVE_NOT = r"have(n't| not)"
YOU = r"\b(you|u)\b"

FORGOT = "|".join([
    r"i ((do|did)(n't| not)|can't|cannot) (recall|remember|think of)",
    r"(hard|difficult) to (recall|think of)",
    r"i forgot",
])

AGREE = "|".join([
    r"\byou('re| are) right\b",
    r"^(?!.*not ).*true$",
    r"^i agree",
    r"i('m| am) with you$",
    r"\byou('re| are) right\b",
    r"^(?!.*(n't|not) ).*true$",
    r"^(?!.*(n't|not) ).*(guess|think) so",
    r"^(?!.*(n't|not) ).*makes sense"
])

DISAGREE = "|".join([
    r"\b(disagree)\b",
    r"don't agree",
    r"(n't|not ) think so",
    r"(n't|not ) make sense",
    r"(n't|not ) true"
])

LIKE = r"(like|love|prefer|recommend|enjoy)"
DISLIKE = r"(dislike|hate|detest|abhor|loathe|despise)"
LETS = rf"(\blet's\b|\blet us\b|\blets\b)"
TELL_ME = fr"(tell me|show me)"
TALK_ABOUT = fr"((talk|chat){MAX_2_WORDS} about)"
WANT_TO = r"(wanna|want to|('d|would) (like|love) to)"
DO_YOU_WANT_TO = fr"do you{MAX_2_WORDS}{WANT_TO}"
CAN_WE = fr"((can|could|shall) (we|i)|how about we|why (dont|don't) (we|you))"
CAN_YOU = fr"((can|could) you)"

CURSE_WORD = r"(dumb|trash|crap|garbage|jerk|stupid|idiot|retarded|deaf|forgetful|not smart|glitching|sick|limited|annoying|boring|terrible|rude|scaring|disappointing|insane|crazy|creepy|irritating|awkward|offensive|filthy|forgetful|terrible)"

PAUSE_VRB = "|".join([
    fr"wait{MAX_1_WORDS}",
    fr"hold( (on|up))?",
    fr"pause({FRONT_MOD}{MAX_1_WORDS})?",
    fr"stop( talking( to me)?)?",
    fr"shut (up|off)",
    fr"be quiet",
])
PAUSE = fr"( ?(just )?(wanna )?( ?{PAUSE_VRB} ?))"
CHAT = fr"( *\b(this|the)?{MAX_1_WORDS}(chat|conversation)\b *)"
LIKE_TO = fr"( *\b(would like|want|like) to\b *)"