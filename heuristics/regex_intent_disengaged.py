from regex_intent_common import *

disengaged_patterns = {
    # Group 1
    "complaint_system_repetition": "|".join([
        fr"{YOU} already (asked|told) me",
        fr"{YOU}.*(mention|said|tol|asked)*(already|before)",
        fr"{YOU} already (mention|said|tol|asked)",
        fr"{YOU}.*(mention|said|told|asked).*(already|before)",
        fr"{YOU}.*keep (saying|asking)",
        fr"why do {YOU}.*keep",
        fr"i informed {YOU}",
        r"i already (asked|told|answered)",
        r"i (told|answered).*(already|before)",
    ]),

    "complaint_system_ignoring_them": "|".join([
        fr"{YOU}('r|re) not listening",
        fr"{YOU} {DO_NOT} (answer|listen)",
        fr"^({YOU} |)answer (my.*|me)",
        r"^listen"
    ]),

    "complaint_system_misunderstanding_them": "|".join([
        fr"\b{YOU}.*({DO_NOT} understand|misunderstood|got it wrong)\b",
        fr"\b{YOU}.*(misunderstood|got it wrong)\b",
        fr"\bi ({DO_NOT}|{HAVE_NOT}) (say|said|ask|asked)\b",
        r"not what i (was (asking|saying)|said|asked)",
        r"i'm (saying|asking)",
    ]),

    "complaint_not_understand_system": "|".join([
        r"confused",
        fr"i ({DO_NOT}|can't) understand.*($|this|that|you{MAX_6_WORDS})$",
        fr"{YOU} got.*wrong",
        r"what are you talking about",
        fr"what (do|did) {YOU} mean",
        r"^what(|\?|!)$",
        fr"^({SORRY}|please |{WH} )?i don't understand{MAX_3_WORDS}(you{MAX_1_WORDS}said{MAX_1_WORDS})?$",
        fr"^can't understand that$",
        fr"^{START}?can you{MAX_1_WORDS}explain{MAX_6_WORDS}$",
        fr"^{START}?explain to me{MAX_6_WORDS}$",
        fr"^{START}?explain{MAX_3_WORDS}$",
        fr"^{START}?explain( (again|yourself))?$",
        fr"^{START}?would you like to explain{MAX_3_WORDS}$",
        fr"^{START}?can you explain that{MAX_3_WORDS}$",
    ]),

    "complaint_curse_system": "|".join([
        fr"{YOU}.*\b{CURSE_WORD}\b",
        fr"how {CURSE_WORD}.*you",
        fr"are you {CURSE_WORD}",
        fr"{CURSE_WORD} question",
        fr"{YOU}.*\b(suck|die)\b",
        fr"{YOU}.*\b(loser|terrible conversationalist)\b",
        fr"{YOU}.*\b(irritate|offend(|ed))\b",
        fr"\b{YOU}.*(make|made|drive) me (mad|nut|uncomfortable)",
        fr"{YOU}.*lost",
        r"how terrible.*you",
        r"what the (hell|heck)",
        fr"what('s| is) wrong with {YOU}",
        r"what happened.*you",
        r"piss",
        r"you.*(go to hell|die)",
        r"kick your ass",
        fr"(hate|don't like) {YOU}",
        fr"angry.*{YOU}",
    ]),

    "express_frustration": "|".join([
        r"\bsigh\b",
    ]),

    # Group 2
    "negative_opinion": "|".join([
        r"\b(suck(|s)|boring|silly|awful|disappointing|terrible|horrible|disturbing|awkward|stupid|fishy|weird|odd|crap)\b",
        r"^(?!.*not ).*(bad)",
        r"\b(fail)\b",
        r"it('s| is) gross",
        r"messed up",
        r"i don't like",
        fr"{DISAGREE}",
        fr"i {DO_NOT}.*{LIKE}",
        fr"i (?!{DO_NOT}).*{DISLIKE}",
        fr"(i('m| am) )?(not|(very )?little){MAX_1_WORDS}(interested in|into|big on)",
        r"(i('m| am) )?not a\b.*\bfan\b",
        r"not my (kind|type) of thing"
    ]),

    "low_interests": "|".join([
        rf"{MAX_1_WORDS} whatever$",
        rf"don't{MAX_1_WORDS} care{MAX_4_WORDS}$",
        rf"not interested"
    ]),

    # Group 3
    "request_topic_change": "|".join([
        r"((can|could) (we|you) (please )?|((\blet's\b|\blet us\b|\blets\b)\s))?((change|get off|stop talking about (the )?)(\b(entertainment|movies|movie|travel|book|books|literature|sport|sports|news|video game.|games|game|music|animals|animal|technology|holiday|obsecure holiday|ted talk)\b( please)?$))|(?<!not to )change (the )?(conversation|(a )?topic|subject)|start over|different (topic|subject)|switch.* topic|^go back (?!to bed)|main menu|next topic|new topic|more topics|new subject|another topic|another subject|other topic|second topic|something else$|anything else$|something different|another thing|move on|^i do next|last topic|some other things",
        r"((can|could) (we|you) (please )?|((\blet's\b|\blet us\b|\blets\b)\s))?((get off|stop( talking)?( about)?|i (don't|do not) (want|wanna) (to )?talk( with you)?( about)?|(i'm|i am) tired of talking( about)?|^i (don't|do not) like|^(i'm|i am) not (into|interested in)) (the )?(\b(entertainment|movies|movie|travel|traveling|book|books|reading|literature|sport|sports|news|video game(s)?|games|game|music|musics|animals|animal|technology|holiday|obsecure holiday|fashion|this|that|it|them)\b( please| anymore)?$))",
        rf"({LETS}\s*{TALK_ABOUT}|(\bi\b)\s*({WANT_TO}|love to|like to)\s*({TALK_ABOUT}|know|ask|learn))(?!( something| anything|$))",
        rf"({CAN_WE}|{CAN_YOU}) ({TALK_ABOUT}|switch to|{TELL_ME}(?! more))",
        rf"^{DO_YOU_WANT_TO} ({TALK_ABOUT}|switch to|{TELL_ME}(?! more))",
        rf"^{TALK_ABOUT}",
        rf"do you (know|have).* (about|on)\b",
        rf"{TELL_ME}(?! more).* about",
        rf"tell me some\b",
        rf"^(i'm|i am) into\b",
        rf"^are there any good\b",
        rf"what is today's news",
    ]),

    "terminate": "|".join([
        r"\bbye\b",
        r"^(\/|)end$",
        r"((don't|dont|not|do not).* (talk|chat) anymore$)",
        r"((let's )?(talk|talking|chat) (about this |it )?later)$",
        r"(^(alexa )?((power off|off|power down|cancel|(i'm|okay) done|leave me alone|dismiss|^can we stop talking)( for a little while)?( please)?$|you can turn off|disconnect|\bi\b.* leave)( please)?$)",
        r"(^(be|alexa) quiet$)",
        r"(cancel.* chat)",
        r"(chat is over)",
        r"(conversation is over)$",
        fr"{MODAL} you (stop|turn off)( now)?( please)?",
        r"(don't|dont|not|do not).* talk to me",
        r"(exit|get out of) social (mode|bot)",

        fr"^thank you good night$",
        fr"^time to go to bed$",
        fr"^(how about )?you need to go to bed$",
        fr"^{START}can we pick up{CHAT}tomorrow$",
        fr"^{START}{MAX_1_WORDS}we{MAX_1_WORDS}talk{MAX_2_WORDS}tomorrow$",
        fr"^{START}go ahead and turn yourself off{MAX_1_WORDS}$",
        fr"^{START}i'll chat with you another time$",
        fr"^{START}i'm calling it a night{MAX_2_WORDS}$",
        fr"(i (think|guess) )?(i|we){GOING_TO_BED}{MAX_2_WORDS}$",
        fr"^{START}(maybe )?it's{MAX_1_WORDS}time (for me )?to go to bed{MAX_1_WORDS}$",
        fr"^{START}(i (guess|think) *)?(it's )?time to go to bed$",
        fr"^{START}it's been really nice talking to you$",
        fr"^{START}(i|we){MAX_2_WORDS}say good night{MAX_1_WORDS}$",
        fr"^{START}(i'm|we're) gonna stop chatting$",

        r"(talk to you tomorrow)$",
        r"(want to stop talking)$",
        r"^((please )?stop talking( please)?)$",
        r"^(conversation over)$",
        r"^(end conversation)$",
        r"^(end this|end (this )?chat|stop conversation)$",
        fr"^{START}(good bye|goodbye)$",
        r"^(let's stop talking)$",
        r"^(let's|let us) call .*a day$",
        r"^(stop chatting)$",
        r"^(stop stop stop)$",
        r"^(turn|shut) off( please)?$",
        r"^(we|you) can stop$",
        r"^can you( please)? stop talking",
        r"^done$",
        r"^have a nice day$",
        r"^i don't want to (talk|listen) to you$",
        fr"^{START}(i )?(need to|have to|gotta) go( have a nice day)?$",
        fr"(?<!do )you need to stop$",
        fr"i (need|have) to leave{MAX_2_WORDS}$",
        fr"i need( you)? to stop( this conversation)?( right)?( now)?( please)?$",
        fr"{START}{MAX_2_WORDS}(?<!do )(i|we|you){MAX_1_WORDS}(?<!don't )(need|have) to stop({MAX_1_WORDS}(chatting|talking)(?! about)| now|$)",
        fr"it's time to end( this conversation)?{MAX_2_WORDS}$",
        r"^let's not talk(?! about)\b",
        r"^no more conversation( please)?$",
        r"^quit$",
        r"^stop (social )?bot",
        r"^stop$",
        r"^turn.*\boff$",
        r"cancel$",
        r"disable social",
        fr"end{CHAT}",
        r"good chatting with you",
        r"got to go$",
        r"i (\b\w*\b )?have to go$",
        r"i gotta go$",
        r"i was finished chatting",
        r"i'm done chatting",
        r"i'm done talking",
        r"i'm done$",
        r"i'm done with you",
        r"it's my bedtime",
        r"quit social",
        r"see you (later|tomorrow|soon|again)$",
        fr"shut up$",
        r"social bot off",
        r"talk( to you)?( again)? later",
        r"we( are|'re) done( here)?",
        fr"(can we|i{LIKE_TO}|how do i) (end|stop){CHAT}",
        fr"^{START}i want you to close$",
        fr"^{START}{MAX_3_WORDS}i have to sleep{MAX_3_WORDS}$",
        fr"^{START}{MAX_3_WORDS}stop talking to me{MAX_3_WORDS}$",
        fr"^{START}{MAX_3_WORDS}i'm going to go{MAX_3_WORDS}$",
    ]),

    # Group 4
    "negative_answer": "|".join([
        r"^(no|not really|nope|nah|na)\b",
        r"^wrong$",
        r"isn't",
        r"\b(none|nothing)",
        r"(i'm|i am) (good|okay)(| thanks)$",
        r"i (did|have)(n't| not)",
        r"\bi don't have (any|a|an|one)\b",
        r"never(?! mind)",
        rf"i {DO_NOT} (watch|have)",
        r"i won't",
        r"my \w+ won't",
        r"not interested",
        r"i hate",
        r"i don't like",
        fr"{FORGOT}",
        fr"{DISAGREE}"
    ]),

    "unsure_answer": "|".join([
        r"i (don't|do not|didn't|did not) (know|remember)",
        r"no idea",
        r"(maybe|probably|possibly)$",
        r"no (comment|opinion)",
        r"not( \w+)* sure",
        r"\b(unsure|uncertain)",
        r"(hard to say|depends)$",
        r"\b((hard|tough) (one|question))",
        fr"{FORGOT}"
    ]),

    "back_channeling": "|".join([
        "^yep$",
        "^yeah$",
        "^k$",
        "^kk$",
        "^ok$",
        "^okk$",
    ]),

    "hesitate": "|".join([
        fr"^(hold on|well|wait|hang on {DURATION})$",
        fr"let me (think|see) ({DURATION}|about it)$",
        fr"i need a break$",
        fr"^{MAX_6_WORDS}(i try to think|i'm thinking){MAX_2_WORDS}( off the top of my head)?$",
        fr"({USER_DO}|{YOU_DO}{PAUSE}) {DURATION}",
        fr"^{MAX_2_WORDS}just {DURATION}",
        fr"^{START}{MAX_3_WORDS}let me see$",
        fr"^i {PAUSE}( ({DURATION}|please))?$",
        r"(be right back|be back soon)",
        fr"i need$",
    ]),
}
