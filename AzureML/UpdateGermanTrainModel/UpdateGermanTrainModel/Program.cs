// This code requires the Nuget package Newtonsoft.Json to be installed.

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace CallUpdateResource
{
    public class AzureBlobDataReference
    {
        // Storage connection string used for regular blobs. It has the following format:
        // DefaultEndpointsProtocol=https;AccountName=ACCOUNT_NAME;AccountKey=ACCOUNT_KEY
        // It's not used for shared access signature blobs.
        public string ConnectionString { get; set; }

        // Relative uri for the blob, used for regular blobs as well as shared access 
        // signature blobs.
        public string RelativeLocation { get; set; }

        // Base url, only used for shared access signature blobs.
        public string BaseLocation { get; set; }

        // Shared access signature, only used for shared access signature blobs.
        public string SasBlobToken { get; set; }
    }

    public class ResourceLocation
    {
        public string Name { get; set; }

        public AzureBlobDataReference Location { get; set; }
    }

    public class ResourceLocations
    {
        public ResourceLocation[] Resources { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            InvokeService().Wait();
        }

        static async Task InvokeService()
        {
            const string url = "https://management.azureml.net/workspaces/8dd9fd82d81f4d44831f1aeda626e14c/webservices/efcc3eaf848243a59ab643ea66cd69d2/endpoints/trainedmodel";
            const string apiKey = "ZpQbrxxgy5QIDp33XBdoZ9GefyvWvjHTWHwOK0Cc6J8hClEJ6TTuVu0/KQslo+587qUlgd06M3sD1ojcc1zDOA=="; // Replace this with the API key for the web service

            var resourceLocations = new ResourceLocations()
            {
                Resources = new ResourceLocation[] {
                    new ResourceLocation()
                    {
                        Name = "Experiment 2 [trained model]",
                        Location = new AzureBlobDataReference()
                        {
                            // Replace these values with the ones that specify the location of the new value for this resource. For instance,
                            // if this resource is a trained model, you must first execute the training web service, using the Batch Execution API,
                            // to generate the new trained model. The location of the new trained model would be returned as the "Result" object
                            // in the response. 
                            BaseLocation = "https://poorvaja1storage.blob.core.windows.net/",
                            RelativeLocation = "germanretrain/trained modelresults.ilearner",
                            SasBlobToken = "?sv=2015-02-21&sr=b&sig=dHgwBsMi5fDqpnN3cOLu%2F2cmj9TUpyFZ6m7agdoSF5I%3D&st=2018-09-19T15%3A19%3A29Z&se=2018-09-20T15%3A24%3A29Z&sp=r"
                        }
                    }
                }
            };

            using (var client = new HttpClient())
            {
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
                using (HttpRequestMessage request = new HttpRequestMessage(new HttpMethod("PATCH"), url))
                {
                    request.Content = new StringContent(JsonConvert.SerializeObject(resourceLocations), System.Text.Encoding.UTF8, "application/json");

                    // WARNING: The 'await' statement below can result in a deadlock if you are calling this code from the UI thread of an ASP.Net application.
                    // One way to address this would be to call ConfigureAwait(false) so that the execution does not attempt to resume on the original context.
                    // For instance, replace code such as:
                    //      result = await DoSomeTask()
                    // with the following:
                    //      result = await DoSomeTask().ConfigureAwait(false)


                    HttpResponseMessage response = await client.SendAsync(request);
                    if (response.IsSuccessStatusCode)
                    {
                        string result = await response.Content.ReadAsStringAsync();
                        Console.WriteLine("Successfully patched path");
                        Console.WriteLine("Result: {0}", result);
                    }
                    else
                    {
                        Console.WriteLine("Failed with status code: {0}", response.StatusCode);
                        string responseContent = await response.Content.ReadAsStringAsync();
                        Console.WriteLine(responseContent);
                    }
                }
            }
        }
    }
}


