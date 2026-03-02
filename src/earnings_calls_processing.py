"""
Created on Wed Feb 25 13:07 2026

@author: Bastian Koch 
"""

# import packages
import re
import string
from pathlib import Path
import nltk
from nltk.corpus import stopwords

# download stopword list on first run (no-op if already cached)
nltk.download("stopwords", quiet=True)
_STOPWORDS = set(stopwords.words("english"))
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Matches speaker header lines, e.g. "ANTHONY KAMIN:", "RAYMOND C. STACHOWIAK:",
# as well as structural labels like "ANSWER_1:", "QUESTION_1:"
_HEADER_RE = re.compile(r'^[A-Z][A-Z0-9\s\.\,_]+:$')


def normalize_text(text: str) -> str:
    """Full normalization pipeline for NLP:

    1. Remove speaker-name header lines (e.g. ``ANTHONY KAMIN:``) and
       structural labels (``ANSWER_1:``, ``QUESTION_1:``).
    2. Remove punctuation.
    3. Remove digits.
    4. Remove English stopwords.
    5. Tokenize by whitespace — output is a single space-separated token string.
    """
    tokens_out = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # drop header/label lines
        if _HEADER_RE.match(stripped):
            continue
        # remove punctuation
        line_clean = stripped.translate(_PUNCT_TABLE)
        # remove digits
        line_clean = re.sub(r"\d+", "", line_clean)
        # tokenize and filter stopwords
        tokens = [t for t in line_clean.split() if t.lower() not in _STOPWORDS]
        tokens_out.extend(tokens)
    return " ".join(tokens_out)

# define working directory relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

input_dir = DATA_DIR / "raw"
output_dir = DATA_DIR / "processed"
segments_dir = output_dir / "Transcripts_nostop"

# ensure output directories exist
segments_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Create an overview file containing information on the call participants
# =============================================================================

# Open the file containing the list of transcripts "list_of_transcripts.csv" to
# open the individual transcripts of the earnings calls
transcript_list_file=open(input_dir / 'list_earnings_calls_group_project_upload.csv','r',encoding="utf-8")
transcript_list_text=transcript_list_file.read()
# Normalise line endings (Windows \r\n -> \n) before splitting
transcript_list_text=transcript_list_text.replace("\r\n","\n").replace("\r","\n")
list_earnings_calls=transcript_list_text.split("\n")
# The last line is empty -> drop it
while list_earnings_calls.count("")>0:
    list_earnings_calls.remove("")
transcript_list_file.close()

# Create an output file
output_csv_file=open(output_dir / 'Transcripts' / 'Overview_Calls.csv','w',encoding="utf-8")
# Write variable names to the first line of the output file
# Columns 1 to 9 should be identical to the columns of the "list_of_transcripts"
# Columns 10 and 11 should contain dummy variables indicating if there is a presentation/Q and A
# Column 12 should display the number of non-corporate call participants
# Column 13 and the following columns should show the names of all corporate
# participants and their positions -> each item should be written in a seperate column
output_csv_file.write(list_earnings_calls[0]+';presentation_found;q&a_found;number_analysts')
# There can be up to 11 corporate participants
for i in range(1,12):
    output_csv_file.write(';manager_'+str(i)+';position_manager_'+str(i))
output_csv_file.write('\n')

# initialize variable to count the maximum number of corporate participants    
max_corporate_participants=0

total_calls = len(list_earnings_calls) - 1
for i in range(1,len(list_earnings_calls)): 
    
    # get the filename of each earnings call
    # read the entire line and split it into columns (i.e., variables)
    # The CSV uses commas as delimiter; filename is at column index 4, year at index 6
    call_information=list_earnings_calls[i].split(",")
    fname = call_information[4].strip('"')
    print(f"Processing {i}/{total_calls}: {fname}", flush=True)
    filename=call_information[4].strip('\"').replace(".txt","")
    year_call=call_information[6].strip('\"')
    
    # open the call transcript (transcripts are organised in year subdirectories)
    call_file=open(input_dir / 'Transcripts' / year_call / (filename+'.txt'),'r',encoding="utf-8")
    call_text=call_file.read()
    # Close the file
    call_file.close() 
    
    # fix a formating issue
    call_text=call_text.replace("&amp;","&")
    
    # =========================================================================
    # Create dummy variables for calls including a management presentation and/
    # or a Q and A session
    # =========================================================================
    
    # reset variables
    presentation_found=0
    qanda_found=0
    
    # search for the beginning of the presentation
    match_presentation=re.search("(?<=\n)={1,}\nPresentation\n-{1,}(?=\n)",call_text)
    if match_presentation:
        presentation_found=1
    
    # search for the beginning of the Q&A
    match_qanda=re.search("(?<=\n)={1,}\nQuestions and Answers\n-{1,}(?=\n)",call_text)
    if match_qanda:
        qanda_found=1
    
    # =========================================================================
    # Identify the number of analysts
    # =========================================================================
    
    # reset variables    
    analysts_text=""
    analysts_list=[]
    number_analysts=0
    
    # search for the part of the text where the analysts are listed
    match_analysts=re.search("(?<=\n)={1,}\nConference Call Participants\n={1,}(?=\n)",call_text)
    # extract the relevant part of the text
    if match_analysts:
        # the list of participants ends where the presentation begins
        if match_presentation:
            analysts_text=call_text[match_analysts.end():match_presentation.start()]
            analysts_text=re.sub("\n{1,}\Z","",analysts_text)
    else:
       # this else condition helps to detect probelms
       print(filename+": NO ANALYSTS FOUND") 
    
    # split the text listing all analysts into a list of analysts
    analysts_list=re.split("\n[ \t]{1,}\*[ \t]{1,}",analysts_text)
    # the first element of the list is just a line feed -> drop it
    while analysts_list.count("")>0:
        analysts_list.remove("")
    
    # the number of analysts corresponds to the length of the list
    number_analysts=len(analysts_list)

    # =========================================================================
    # Identify the corporate managers and their positions
    # =========================================================================
    # reset variables
    managers_found=0
    managers_text=""
    manager_name_list=[]
     
    # search for the beginning of the text listing the managers
    match_managers=re.search("(?<=\n)={1,}\nCorporate Participants\n={1,}(?=\n)",call_text)   
    # get the names of the corporate participants and their position
    if match_managers:
        managers_found=1
        # identify the end of the managers list
        if match_analysts:
            managers_text=call_text[match_managers.end():match_analysts.start()]
            managers_text=re.sub("\n{1,}\Z","",managers_text)
        else:
            print(filename+": NO Q AND A PART FOUND")
            # there is no Q and A and hence no conference call participants
            # in this case the beginning of the presentation marks the end
            # of the manager list
            if match_presentation:
                managers_text=call_text[match_managers.end():match_presentation.start()]
            else:
                print(filename+": NO PRESENTATION FOUND")
    else:
        # like before, this else condition helps us to identify problems
        print(filename+": BEGINNING OF MANAGERS LIST NOT FOUND")   
    
    # split the extracted text into a list
    managers_list=re.split("\n[ \t]{1,}\*[ \t]{1,}",managers_text)
        
    # the first element of the list is just a line feed -> drop it
    while managers_list.count("")>0:
        managers_list.remove("")

    # identify the maximum number of corporate particpants
    if len(managers_list)>max_corporate_participants:
        max_corporate_participants=len(managers_list)

    # write the call information to the output file
    output_csv_file.write(list_earnings_calls[i]+";"+str(presentation_found)+";"+\
    str(qanda_found)+";"+str(number_analysts)) 
        
    manager_name=""
    manager_position=""
    
    if len(managers_list)>0:
        for j in range(len(managers_list)):
            # depending on how you split the text of corporate participants,
            # one element of your list could contain the name of the mangager 
            # in the first line and their position in the second line.
            # ADJUST THE FOLLOWING COMMANDS IF YOU USED A DIFFERENT SPLIT.
            
            # split each element of the list of corporate participants further 
            # into name and position
            manager_entry=managers_list[j]
            manager_entry_parts=re.split("\n[ \t]{0,}",manager_entry)
            
            manager_name=manager_entry_parts[0]
            # for part b) of the problem it is helpful to have a list of all
            # manager names. With this list, we can identify whether a statement
            # comes from a managers (-> answer) or from an analyst (-> question)
            manager_name_list.append(manager_name)
            # there are a few instances where the position of the manager is 
            # not specified
            if len(manager_entry_parts)>1:
                manager_position=manager_entry_parts[1]
                # Like before, the template assumes a very specific type of split here
                # So depending on your approach, you might need to change the commands below.
                # the position is just the text part after " - "
                # For example
                # Bank of America Corporation - CEO
                # the position is "CEO"
                match_position=re.search("(?<= - )[^\n]{1,}",manager_position) or\
                re.search("(?<=\A- )[^\n]{1,}",manager_position)
                # The second regex after the or is needed to handle cases
                # in which the transcript does not show a company name for the
                # manager.
                # you find an example in the file "earnings_call_DIS_2005-05-11.txt"
                # For the first manager, the company name is missing
                # Wendy Webb
                # - SVP IR and Shareholder Services
                # for the other managers, the normal regex works
                # Michael Eisner
                # Walt Disney - CEO, Director
                # To avoid missing data, we allow for both formats.
                
                if match_position:
                    manager_position=match_position.group(0)
                else:
                    # the position is not displayed in the usual format
                    manager_position="NN"
                    #print(filename+" :"+manager_position)
            else:
                manager_position="NN"
                
            # write the manager names and positions to the output file
            output_csv_file.write(";"+manager_name+";"+manager_position.replace("\n",""))
        
    output_csv_file.write("\n")     
    

    # =========================================================================
    # PART B: Extract the call segments
    # =========================================================================        
    presentation_text=""
    question_text=""
    answer_text=""
    # create an empty text variable where you can store the text parts that you remove 
    # while cleaning the transcript texts
    dropped_text=""
    
    # =========================================================================
    # Identify the presentation    
    # =========================================================================
    # the presentation ends where the Q and A part begins
    # match_presentation and match_qanda have been defined before.
    if match_presentation:
        if match_qanda:
            presentation_text=call_text[match_presentation.end():match_qanda.start()]
        else:
            # there is no Q and A
            presentation_text=call_text[match_presentation.end():]           

    # =========================================================================
    # Drop operator/editor statements
    # =========================================================================
    # we are now preprocessing the presentation further
    match_operator=re.search("\n{0,}-{0,}\n(Operator|Editor) {1,}\[[0-9]{1,3}\]\n-{1,}",presentation_text)
    while match_operator:
        match_operator_start=match_operator.start()
        # search for the end of the operator statement -> beginning of the next speaker
        # Hint: search only after the beginning of the operator statement
        # Hint 2: remember to keep track of your coordinates (.start() and .end())
        match_operator_end=re.search("\n-{1,}\n[^\[\n]{1,} {1,}\[[0-9]{1,3}\]\n-{1,}",presentation_text[match_operator.end():])
        if match_operator_end:
            operator_text=presentation_text[match_operator_start:match_operator.end()+match_operator_end.start()]
        else:
            # the presentation ends with an operator statement, i.e., there is no further speaker
            operator_text=presentation_text[match_operator_start:]
            
        #print(operator_text+"\n")
        dropped_text=dropped_text+"OPERATOR: "+operator_text+"\n"
        
        presentation_text=presentation_text.replace(operator_text,"")
        
        # check whether there is another match
        match_operator=re.search("\n{0,}-{0,}\n(Operator|Editor) {1,}\[[0-9]{1,3}\]\n-{1,}",presentation_text)
    
    # =========================================================================
    # Drop information on the speakers, e.g.,
    # -------------------------------------------------------------------------
    # Deborah Crawford,  Facebook, Inc. - Director of IR    [2]
    # -------------------------------------------------------------------------
    # =========================================================================
    
    match_speaker=re.search("\n-{1,}\n[^\[\n]{1,} {1,}\[[0-9]{1,3}\]\n-{1,}\n",presentation_text)
    while match_speaker:
        # the task is similar to the deletion of the operator statement but be careful
        # to only remove the speaker name but NOT the text of the speaker.
        #print(match_speaker.group(0))
        presentation_text=presentation_text.replace(match_speaker.group(0),"",1)
        # check whether there is another speaker name
        match_speaker=re.search("\n-{1,}\n[^\[\n]{1,} {1,}\[[0-9]{1,3}\]\n-{1,}\n",presentation_text)
    
    # =========================================================================
    # Remove technical remarks   
    # =========================================================================
    # sometimes there are technical remarks like "(inaudible)", "(corrected by company after the call)",
    # or "(technical difficulty)" -> drop those
    match_technical_remarks=re.search("(?<=\W)\([^\)]{1,}\)(?=\W)",presentation_text)
    if match_technical_remarks:
        print("\nThis is a technical remark:\n"+match_technical_remarks.group(0))
        presentation_text=presentation_text.replace(match_technical_remarks.group(0),"")
        # check whether there is another match
        match_technical_remarks=re.search("(?<=\W)\([^\)]{1,}\)(?=\W)",presentation_text)
    # there are several ways to approach this editing step (e.g., re.sub())
    
    # formatting
    # delete multiple line breaks
    presentation_text=re.sub("\n{2,}","\n",presentation_text) 
    # delete whitespaces/tabs at the beginning of a line
    presentation_text=re.sub("\A[ \t]{1,}","",presentation_text)
    presentation_text=re.sub("\n[ \t]{1,}","\n",presentation_text)
    
    # normalise: remove headers, punctuation, digits, stopwords; whitespace-tokenise
    presentation_text = normalize_text(presentation_text)

    # write the presentation text to an output file
    output_file_presentation=open(segments_dir / (filename+'_presentation.txt'),"w",encoding='utf-8')
    output_file_presentation.write(presentation_text)
    # Close file
    output_file_presentation.close()
    
    # =========================================================================
    # identify questions and answers
    # =========================================================================
    
    # reset variables
    qanda_text=""    
    qanda_list=[]
    
    if match_qanda:
        # the text of the Q and A goes from the beginning of the Q&A (see above)
        # to the end of the document
        qanda_text=call_text[match_qanda.end():]
        
        # formatting (see commands as above when creating the presentation files)
        # delete multiple line breaks
        qanda_text=re.sub("\n{2,}","\n",qanda_text)
        # delete whitespaces/tabs at the beginning of a line
        qanda_text=re.sub("\A[ \t]{1,}","",qanda_text)
        qanda_text=re.sub("\n[ \t]{1,}","\n",qanda_text)
        
        # =====================================================================
        # Remove technical remarks   
        # =====================================================================
        # same editing operations as in the presentation
        match_technical_remarks=re.search("(?<=\W)\([^\)]{1,}\)(?=\W)",qanda_text)
        if match_technical_remarks:
            #print(match_technical_remarks.group(0))
            presentation_text=presentation_text.replace(match_technical_remarks.group(0),"")
            # check whether there is another match
            match_technical_remarks=re.search("(?<=\W)\([^\)]{1,}\)(?=\W)",qanda_text)

        # split the Q and A part by speaker
        qanda_list=re.split("\n-{1,}\n(?=[^\[\n]{1,}\[[0-9]{1,3}\]\n-{1,}\n)",qanda_text)
        
        # create dummy variables that are 1 if the text part is a managment answer,
        # question, or an operator statement and zero else
        answer=0
        question=0
        operator=0
        
        # create variables that count the number of the questions/answers
        answer_counter=1
        question_counter=1
        
        speaker_text=[]
        
        # go over all parts of your qanda_list, i.e., process the text of all speakers
        for k in range(0,len(qanda_list)):
            # identify the name of the speaker to see whether he/she is a manager
            # you need to separate the speaker's name from what he/she is saying
            speaker_text=re.split("(?<=\])\n-{1,}\n",qanda_list[k])
            # skip entries that don't match the expected speaker format
            if len(speaker_text) < 2:
                continue
            speaker_info=speaker_text[0]
            speaker_name=speaker_info.split(", ")[0]
            speaker_name=speaker_name.replace("\n","")
            #print(speaker_name)
            text=speaker_text[1]
            
            # check whether the speaker name is in the manager list, i.e., whether
            # we have an answer by a corporate participant.
            if speaker_name in manager_name_list:
                # it is a management answer
  
                if k==0:
                    # the first element of the Q and A list (k==0) should be the operator
                    # statement and the second should be an analyst question
                    # If that is not the case there is an opening statement by
                    # the management -> drop it because it is not an answer
                    #print(text+"\n")
                    #write all statements that are dropped to an output file
                    dropped_text=dropped_text+speaker_name.upper()+": "+text+"\n"
                
                elif answer==0 and question_counter>answer_counter:
                    # the previous text was not an answer and we have more
                    # questions than answers. --> add the text of the speaker
                    # to the text of all answers
                    answer_text=answer_text+"ANSWER_"+str(answer_counter)+":\n"+speaker_name.upper()+":\n"+text+"\n"
                    # this statement is an answer but not a question nor an operator statement
                    answer=1
                    answer_counter=answer_counter+1
                    question=0
                    operator=0
                
                else:
                    # the last list_element was an answer
                    # we must add this statement also to the answer document
                    # but not as a new answer but as an addon to the previous answer
                    answer_text=answer_text+"\n"+speaker_name.upper()+":\n"+text+"\n"
                    question=0
                    operator=0
            
            # if the if-condition form above is false, it can be the operator
            # or an analyst question.
            # Check whether it is an operator statement.
            # Note that sometimes there is an "Editor" which is the same as an "Operator."
            elif speaker_name.startswith("Operator") or speaker_name.startswith("Editor"):
                # it is an operator statement
                operator=1
                # we do not include operator statements -> drop them
                dropped_text=dropped_text+"OPERATOR: "+text+"\n"
            
            else:
                # if the speaker is neither a manager nor an operator, he/she must
                # be an analyst
                if question==0 or k==0:
                    # the last element was an answer (i.e., question=0) or
                    # the Q and A session starts with a question (i.e., k=0)
                    
                    # Often the analysts thank the management for answering their
                    # questions -> drop these statements
                    if k==len(qanda_list)-1:
                        # if the Q and A ends with an analyst statement, drop it as there is no reply by the management
                        dropped_text=dropped_text+speaker_name.upper()+": "+text+"\n"
                    # look for common "thank you" phrases
                    elif ((text.count("thank")>0 or text.count("Thank")>0) and len(text)<100) or\
                        (len(text)<50 and not text.count("?")) or\
                        (k<len(qanda_list)-1 and (qanda_list[k+1].startswith("Operator") or qanda_list[k+1].startswith("Editor"))):
                        #print(text+"\n")
                        # write all statements that are dropped to an output file
                        dropped_text=dropped_text+speaker_name.upper()+": "+text+"\n"
                    # if it is not a thank you statement and if it is neither the
                    # last statement in the call, it is a regular question.
                    else:    
                        # write the quetion to the question file
                        question_text=question_text+"QUESTION_"+str(question_counter)+":\n"+speaker_name.upper()+":\n"+text+"\n"
                        # this statement is a question but not an answer nor an operator statement
                        question=1
                        question_counter=question_counter+1
                        answer=0
                        operator=0
                
                else:
                   # the last list_element was a question (question==1)
                   # This means that a statement is a direct follow up to the previous question
                   # add it without increasing the question counter by 1.
                   #print(filename+"\n"+speaker_info+text)
                   question_text=question_text+":\n"+speaker_name.upper()+":\n"+text+"\n"
                   question=1
                   answer=0
                   operator=0
                    
                   
        # check whether there are as many questions as answers
        if answer_counter == question_counter:
            pass
        else:
            print("The number of questions and answers does not match: "+filename)
        
        
    # when we have processed all statements, we can write the question text,
    # answer text, and dropped text to the corresponding output files.
    output_file_answers=open(segments_dir / (filename+'_answers.txt'),"w",encoding='utf-8')
    output_file_questions=open(segments_dir / (filename+'_questions.txt'),"w",encoding='utf-8')
    output_file_answers.write(normalize_text(answer_text))
    output_file_questions.write(normalize_text(question_text))
    output_file_dropped_text=open(segments_dir / (filename+'_deleted_text.txt'),"w",encoding='utf-8')
    output_file_dropped_text.write(dropped_text)
    # close all three files.
    output_file_answers.close()
    output_file_questions.close()        
    output_file_dropped_text.close()

print("The maximum number of corporate participants is: "+str(max_corporate_participants))
 
print("Task completed.")
# Close files
output_csv_file.close()
