# The simple RAG with Llamaindex, OpenAI and Robocorp tutorial.

Build RAG in an hour! This tutorial walks throught he build of a simple RAG agent using [OpenAI](https://openai.com/), [Llamaindex](https://www.llamaindex.ai/) and running it on Robocorp. Here's what it will do:

> Hey I have these questions about these documents. Help me!

![emails](https://github.com/tonnitommi/email-chat-docs/assets/40179958/a467d17b-705b-4c85-8b35-a082917817c5)

- Workflow is triggered via emails where body has some questions about the PDF files as attachments. The email triggering is a great feature on Robocorp Control Room, and is available for every Python workflow.
- Looks at email body, and extracts clear questions out of it.
- Creates embeddings and the vector index on the fly from the documents found from attachments.
- Finds relevant contexts or "nodes" from the index
- Feeds them to the final step for answering the original questions.
- Sends the results back as an email.


## Get your setup ready

Robocorp is a great way of running these types of Python workloads, with zero infrastructure setup needed!

Still, a few things needs to be in place for things to work. Follow these steps and you are good to go!

1. Make sure you have [VS Code](https://code.visualstudio.com/download) installed, and the [Robocorp Code extension](https://marketplace.visualstudio.com/items?itemName=robocorp.robocorp-code) activated. Then clone the [repo](https://github.com/tonnitommi/email-chat-docs).
2. Robocorp creates fully isolated Python environments based on the [conda.yaml](conda.yaml) configuration files, so just remember no `pip install` of anything needed.
3. Some credentials are needed for things to run: OpenAI and Gmail (optional). You can have them in the Robocorp Control Room and connect the VS Code to your workspace, but here we assume that you are not there yet. We'll use the local secrets file for now.
   1. Remove the `_EXAMPLE` from the name of the [secrets.json_EXAMPLE](secrets.json_EXAMPLE) file.
   2. Modify the file to have your own OpenAI API key.
   3. (OPTIONAL) If you want the automation to send replies as emails, add your own gmail address, and app password. [Here's how to generate the password.](https://support.google.com/mail/answer/185833?hl=en). If you don't want to do that, the workflow is just printing the final responses to the console.
   4. (OPTIONAL) If you chose to add your gmail credentials, then also edit the example work item that you can use for testing locally. Look for from/email address in the [work-items.json](devdata/work-items-in/example-email-big/work-items.json) file. We'll show later how you can run things locally to test!

With these done, you are ready to go!

## What's in the code

Here are some of the key parts of the code explained. Check the [code](tasks.py) for more details, it has comments explaining the logic behind the steps.

```py
openai_credentials = vault.get_secret("OpenAI")
openai.api_key = openai_credentials["key"]
```

The example uses Control Room Vault for storing the secrets. When developing and running the code locally, the Robocorp Code extension keeps you conencted to the Vault, and let's you securely access the credentials. When running the code through Control Room in production, these are provided by the Control Room to the execution environment.

```py
item = workitems.inputs.current
...
email = item.email()
paths = item.get_files("*.pdf", DATA_FOLDER)
```

Work Items is one core concept of Robocorp platform. One Work Item contains one atomic piece of work, and the Work Data Management implements a queue or pool of work to be completed. These lines pull the contents of the current work item, take the email contents our of them, and extracts all the paths of .pdf files from attachments.

Another convenience of the platform is that it offers email triggering. It means that each process has it's own email address, and when that address receives an incoming email, a process execution is started and the email content is automatically a work item. Saves a ton of code that would otherwise be needed!

```py
response = openai.ChatCompletion.create(
   model="gpt-4",
   messages=[
      {"role": "system", "content": "Your are an assistant helping extract structured data from the messages."},
      {"role": "user", "content": QUESTIONS_EXTRACT_PROMPT + email.text}
   ]
)
```

This is the first time calling OpenAI, extracting the questions from the body of user's incoming email. It uses `gpt-4`, but most likely smaller models would work out just fine, too.

💡 Improvement idea: the prompts are inline with the code in this example, but in real life scenarious you'd probably want to separate them away from the code. Have a look at the [example which put's them in to Control Room Asset Storage](https://robocorp.com/portal/robot/tonnitommi/example-prompt-template-assets).


```py
def create_index(folder):
   reader = SimpleDirectoryReader(folder)
   docs = reader.load_data()
   index = VectorStoreIndex.from_documents(docs)
```

Llamaindex shines here! This is how easy it is to create the vector index from a folder full of documents. Three lines. 🤯


```py
query_engine = index.as_query_engine()
...
response = query_engine.query(line)
```

Once the query engine is created out of the vector index, it's just one line of code to find the relevant pieces of content that match with `line`, which basically is one question from the user's email. This is basically the "R" of RAG. Now, in reality you'll probably need to put more effort and customise the query engine to meet exactly the needs of your use case. But to get started, this simple approach is great.

```py
for node in response.source_nodes:
   if node.score > THRESHOLD:
      final_prompt = final_prompt + "\n\n***CONTEXT***\n" + node.text + f"\n***SOURCE:*** File: {node.metadata['file_name']}, page{node.metadata['page_label']}\n"
```

After the "nodes" are found from all the content, this part of code constructs the final prompt that will have the contexts, as well as references to the sources so that they can be included in the answer that goes back to the user. It also uses a configurable `THRESHOLD` that determines how good hits are used and when not.

```py
final_response = openai.ChatCompletion.create(
   model="gpt-4",
   messages=[
      {"role": "system", "content": "Your are an assistant helping to answer user's question based on the information found by other AI Assistants."},
      {"role": "user", "content": final_prompt}
   ]
)
```

The prompt to LLM for generation of the final answer based on the contexts found earlier is also using `gpt-4`. A simple "vanilla" call to OpenAI just to keep things clear.

```py
notifier.notify_gmail(
   message=body,
   to=email.from_.address,
   username=gmail_credentials["email"],
   password=gmail_credentials["email-app-password"],
   subject="Message from a friendly bot")
```

In the end, turning againg to Robocorp's capabilities using [RPA Framework's Notifier library](https://robocorp.com/docs/libraries/rpa-framework/rpa-notifier), that allows super easy way to send emails using gmail. You can easily customize this to be Slack messages or what ever!

💡 Improvement idea: The message is now plain text email and not a reply to the original message. You could make it look much better! 💅

## Running it

**Run locally**: When running the code from VS Code, remember to start the execution from Robocorp Code extension. The repo contains an [example Work Item](/devdata/work-items-in/example-email-big/work-items.json) that has the correctly formed email and a related attachment file.

**Deploy to Robocorp Control Room**: To ship the code to be ran in the Cloud, you can either upload it directly from VS Code to Control Room (Command Palette - Upload to Control Room), or push through GitHub or GitLab. Remember, after the code is uploaded to the Control Room, you should create a process out of it.

**Running in cloud**: Just send an email to your process, and wait for the reply to come back! 🤖



