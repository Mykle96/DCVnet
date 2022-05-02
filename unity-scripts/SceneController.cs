using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.IO;
using UnityEngine.SceneManagement;
using static System.Math;

using Newtonsoft.Json;

/*
This script generates synthetic data using unity. Containers and small boxes are generated at random
locations within the camera view. The camera changes position every 5th frame. In each frame, data of 
the containers are stored in a .txt files. 
*/


public class SceneController : MonoBehaviour
{
    /*
    synth is the imported scipt ImageSynthesis which is used to store masked and regluar images
    that are generated.
    prefabs is a list of GameObjects, and are the objects generated during the running of the script. 
    minObjects and maxObjects are the minimum and maximum generated objects at each frame. 
    */
    public ImageSynthesis synth;
    public GameObject[] prefabs;
    public int minObjects = 1;
    public int maxObjects = 4;
    
    /*
    Instead of generating new objects at each frame, previously generated objects are utilized 
    and stored in a "pool". The script ShapePool.cs handles generation and storage of gameobjects. 

    initialContainers are the containers placed in the scene BEFORE running the script, meaning that 
    containers could be placed at random in the scene and the position data of them will be stored, 
    when they are visible to the camera. 
    */

    private ShapePool pool;
    private int frameCount = 1;
    private int imgCount = 1;
    private Camera maincam;
    private List<GameObject> initialContainers;
    private int frameWidth = 600;
    private int frameHeight = 600;
    private string filepath = "dataset/train";
    private Dictionary<string,Vector3> cornerCoordinates;
    private string[] CornerLabel = {"fbl","fbr","ftl","ftr","bbl","bbr","btl","btr","center"};
    private Vector3[] localCoord = { 
        new Vector3(0.0f, 0.0f, 0.0f), //fbl
        new Vector3(0.0f, 0.0f, -3.0f), //fbr
        new Vector3(0.0f, 3.0f, 0.0f),  //ftl
        new Vector3(0.0f, 3.0f, -3.0f), //ftr
        new Vector3(9.0f, 0.0f, 0.0f), //bbl
        new Vector3(9.0f, 0.0f, -3.0f), //bbr
        new Vector3(9.0f, 3.0f, 0.0f), //btl
        new Vector3(9.0f , 3.0f, -3.0f), //btr
        new Vector3(4.5f,1.5f,-1.5f) //Center 
        };

    private Vector3[] localPlaneCoords = { 
        //new Vector3(4.5f,1.5f,-1.5f), //Center 
        new Vector3(0.0f, 1.5f, -1.5f), //front
        new Vector3(4.5f, 1.5f, 0.0f), //left
        new Vector3(4.5f, 1.5f, -3.0f), //right 
        new Vector3(4.5f, 0.0f, -1.5f), //bottom
        new Vector3(9.0f, 1.5f, -1.5f),  //back 
        new Vector3(4.5f, 3.0f, -1.5f)  //top
        };
    
    
    void Start()
    /*
    This function does the following: 
    Defines the variable maincam(the only camera used in the scene).
    Creates a textfile. 
    Stores initial containers in a list of GameObjects
    Creates the pool of gameobjects that will be placed at random in the scene. 
    */
    {   
        initialContainers = new List<GameObject>(); 
        maincam = GetMainCamera();

        //Ta bort disse?
        //string cameraPos = JsonUtility.ToJson(maincam.transform.position);
        //string cameraRot = JsonUtility.ToJson(maincam.transform.rotation.eulerAngles);
        //File.WriteAllText(filepath+"/pose.txt", cameraPos+"\n"+cameraRot+"\n");

        //File.WriteAllText(filepath+"/pose.txt", filepath+"\n");
        SaveInitialContainers();
        pool = ShapePool.Create(prefabs);
    }
    
   
    void Update()
    /*
    This function runs at each frame, and does the following:
    Every fifth frame, the maincamera will be placed at a new position. 
    The data generated will be stored in three different folder(train, val, test),
    and after a set ammount of frames the script will stop.
    At every frame GenerateRanom() will generate new object within the camera view, and will check 
    if any initial containers are placed within the camera view. 
    The images are stored using synth.save and the frameCount is increased.
    */
    {       
        
        
        if (frameCount % 5 ==0){
            if (frameCount % 25 == 0){
            ChangeMainCameraPosition();
            }
            string filename = imgCount.ToString();
        
            //Change filepath 
            if(imgCount == 350){
                filepath = "dataset/val";
                //File.WriteAllText(filepath+"/pose.txt", filepath+"\n");
            }else if (imgCount == 425){
                filepath = "dataset/test";
                //File.WriteAllText(filepath+"/pose.txt", filepath+"\n"); ;
            }
            if (imgCount == 500){
                UnityEditor.EditorApplication.isPlaying = false;
            }
            
            File.WriteAllText(filepath+$"/{imgCount}.txt", "");
            GenerateRandom();
            CheckForInitialContainers();
            synth.Save(filename, frameWidth, frameHeight, filepath);
        
            imgCount++;
        }
        
        frameCount++;

    }
   


    private Camera GetMainCamera(){
        /*
        Function to retrieve main camera used in scene
        
        return:
            Main camera of scene as a Camera object
        */
        Camera maincam = GameObject.Find("Main Camera").GetComponent<Camera>();
        return maincam;
    }
    
    private void ChangeMainCameraPosition(){
    /*
    Function for changing the camera positon.
    Generates a random position in the interval placed in the Random.Range() function
    Quaternion.Eurler(x,y,x) returns a quaternion with the rotation around the x, y, z -axis. 
    The postion and rotation of the main camera is then changed.
    */
        //Position
        float newX, newY, newZ;
        newX = Random.Range(-50.0f, 50.0f);
        newY = Random.Range(30.0f, 50.0f);
        newZ = Random.Range(-170.0f, 30.0f);
        Vector3 newPos = new Vector3(newX, newY, newZ);
        //Rotation
        var newRot = Quaternion.Euler(Random.Range(-15.0f, 40.0f),Random.Range(-30.0f, 30.0f),Random.Range(-3.0f, 3.0f));
        maincam.transform.position = newPos;
        maincam.transform.rotation = newRot;
        
    }

    void SaveInitialContainers(){
        /*
        This function iterates through the root objects in the scene and stores the 
        containers placed in the scene prior to running the script in a list called initialContainers()
        The containers are stored as GameObjects
        */
        List<GameObject> rootObjects = new List<GameObject>();
        Scene scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        scene.GetRootGameObjects( rootObjects );
        print($"Number of root objects: {rootObjects.Count}");

        for (int i = 0; i < rootObjects.Count; ++i){
            GameObject gameObject = rootObjects[ i ];

            if (gameObject.name.Contains("Container")){
                
                //MeshRenderer rend = gameObject.GetComponentInChildren<MeshRenderer>();
                initialContainers.Add(gameObject);
                //saveGameObject(gameObject, maincam.transform.position, rend);
            }
        }
    }


     private Dictionary<string,Vector3> FindCornersInScreen(GameObject gameObj){
        
        bool tmp = false;
        
        Dictionary<string,Vector3> cornerCoord = GetCornerCoordinates(gameObj);
        List<string> keys = new List<string>();
        
        foreach (var item in cornerCoord){
            Vector3 intContScreenPos = maincam.WorldToViewportPoint(item.Value);
            if ((intContScreenPos.x*(1-intContScreenPos.x)>=0) && (intContScreenPos.y*(1-intContScreenPos.y)>=0)){

                tmp = true;   
            }else{
                keys.Add(item.Key);
            }
        }
        
        foreach (string key in keys){
            cornerCoord.Remove(key);
        }
        return cornerCoord;
    }
    
    void CheckForInitialContainers(){
        /*
        The function iterates through all initial containers.
        The intContScreenPos is a Vector3 with (x,y,z) where (x,y) is screen coordinates where 0,0
        is bottom left and 1,1 is top right corner of the screen. z is the world units from camera. 
        This if statement checks is the (x,y) coordinates returned are 0<=(x,y) <=1, meaning that the object 
        is within the camera view. If so, saves the gameObject.
        */
        
         for (int j = 0; j < initialContainers.Count; ++j){
            GameObject intCont = initialContainers[ j ];
            
            //TA bort?
            MeshRenderer intContRend = intCont.GetComponentInChildren<MeshRenderer>();
            //

            bool tmp = false;
            
            Dictionary<string,Vector3> cornerCoord = GetCornerCoordinates(intCont);
            List<string> keys = new List<string>();

            foreach (var item in cornerCoord){
                Vector3 intContScreenPos = maincam.WorldToViewportPoint(item.Value);
                if ((intContScreenPos.x*(1-intContScreenPos.x)>=0) && (intContScreenPos.y*(1-intContScreenPos.y)>=0)){
                    tmp = true;   
                }else{
                    keys.Add(item.Key);
                }
            }
            
            foreach (string key in keys){
                cornerCoord.Remove(key);
            }
           
            if(tmp){
                saveGameObject(intCont, maincam.transform.position, intContRend, cornerCoord);
            }
            //
            /*
            Vector3 intContScreenPos = maincam.WorldToViewportPoint(intContRend.bounds.center);
            
            if ((intContScreenPos.x*(1-intContScreenPos.x)>=0) && (intContScreenPos.y*(1-intContScreenPos.y)>=0)){
                saveGameObject(intCont, maincam.transform.position, intContRend);
            }*/
    }}
    
    void saveGameObject(GameObject gameobj, Vector3 cameraPos,  MeshRenderer renderer, Dictionary<string,Vector3> worldCornerCord){
        /*
        Function saves data of the container(GameObject) to a .txt file. 

        Args:
            gameobj: The container whose data is going to be written to the .txt file
            camerPos: The position of the camera where the containers was detected
            renderer: the meshRenderer of the container(gameObject), used to find the object(container) center
        */
        //Distance from object center to camera
        float dist = Vector3.Distance(renderer.bounds.center, cameraPos);

        //Gjør om til koordinatsystem hvor kamera alltid er i origo. 
        Vector3 containerCamPosition = gameobj.transform.position - cameraPos;
        Vector3 cameraCamPos = cameraPos - cameraPos;
        
        //Object center
        Vector3 center = renderer.bounds.center;
        Vector3 camCenter = center - cameraPos;

        //Screen coordinates of object center and position
        Vector3 screenCoord = maincam.WorldToViewportPoint(center);
        Vector3 screenPosCoord = maincam.WorldToViewportPoint(gameobj.transform.position);
        

        //Formating dictionary to string TO-DO: WRITE TO FUNCTION 
        string stringCornerCord = "{";
        foreach (var item in worldCornerCord){
            
            string valueString = $"\"x\":{item.Value.x.ToString()},\"y\":{item.Value.y.ToString()},\"z\":{item.Value.z.ToString()}";

            stringCornerCord += $"\"{item.Key.ToString()}\":"+"{"+$"{valueString}" + "},";
           
        }
        stringCornerCord = stringCornerCord.Remove(stringCornerCord.Length-1,1);
        stringCornerCord += '}';

        string stringScreenCornerCord = "{";
        foreach (var item in worldCornerCord){
            
            Vector3 tmp = maincam.WorldToViewportPoint(item.Value);
            
            string valString = $"\"x\":{tmp.x.ToString()},\"y\":{tmp.y.ToString()},\"z\":{tmp.z.ToString()}";

            stringScreenCornerCord += $"\"{item.Key.ToString()}\":"+"{"+$"{valString}" + "},";
           
        }
        stringScreenCornerCord = stringScreenCornerCord.Remove(stringScreenCornerCord.Length-1,1);
        stringScreenCornerCord += '}';

        /* This can be used to check which sides of the container is
        faced towards the camera 
        //print dot-product 
        Vector3 contCcenter = new Vector3(4.5f,1.5f,-1.5f);
        Vector3 dotCenter = gameobj.transform.TransformPoint(contCcenter);
        foreach (var item in localPlaneCoords){

            Vector3 planePos = gameobj.transform.TransformPoint(item);
        
            //Vektor fra center av container til plan
            //sluttpunkt - startpunkt -> container-kamerapos
            Vector3 b = planePos - dotCenter;

            //Vektor fra kamera til plan 
            Vector3 k = planePos - cameraPos;

            //dot product mellom vector beregnet og vektor fra kamerapos til planpunkt
            print($"Dot product: {Vector3.Dot(k,b)}, ImageCount {imgCount}");
        }*/ 
       
        SaveObject saveObject = new SaveObject{
                label = "container",
                img_num = imgCount.ToString(),
                containerCameraPos = containerCamPosition,
                cameraCameraPos = cameraCamPos,
                containerWorldPos = gameobj.transform.position,
                cameraWorldPos = cameraPos,
                centerDistFromCamera = dist,
                containerCameraCenter = camCenter,
                containerWorldCenter = center,
                containerWorldRot = gameobj.transform.rotation.eulerAngles,
                cameraWorldRot = maincam.transform.rotation.eulerAngles,
                screenCenterCoordinates = screenCoord,
                screenPositionCoordinates = screenPosCoord,
                worldCornerCoordinates = stringCornerCord,
                screenCornerCoordinates = stringScreenCornerCord
            };
           string json = JsonUtility.ToJson(saveObject);
           
           //File.AppendAllText(filepath+$"/{imgCount}.txt", json.Replace("'","\"")+"\n" );
           File.AppendAllText(filepath+$"/{imgCount}.txt", json+"\n" );
    }
    
    void GenerateRandom(){
        /*
        Main function that generates random objects in the scene. Runs at every frame.
        */

        //Removes all previously generated gameobjects 
        pool.ReclaimAll();

        //Generates a random number of objects between min and maxobjects.
        int objectsThisTime = Random.Range(minObjects, maxObjects);
        for (int i = 0; i < objectsThisTime; i++){
        //Pick out a prefab 

            //int prefabIndx = Random.Range(0, prefabs.Length);
            int prefabIndx;
            int tmpInt = Random.Range(0,10);
            if (tmpInt == 1){
                prefabIndx = 1;
            }else{
                prefabIndx = 0;
            }
            GameObject prefab = prefabs[prefabIndx];
        
 
        
        //Predefined size of container, set larger than acutal size due to center of container being front left bottom corner 
        Vector3 size = new Vector3(9.0f,3.0f,3.0f); 
        //Position in front of camera 
        Vector3 screenPosition = maincam.ViewportToWorldPoint(new Vector3(Random.Range(0.2f,0.8f), Random.Range(0.2f,0.8f), Random.Range(20.0f,100.0f)));
        Vector3 newPos = screenPosition;
        
        //Rotation
        //var newRot = Quaternion.Euler(Random.Range(0,40),Random.Range(0,365),Random.Range(0,20));
        var newRot = Quaternion.Euler(0,90,0);

        
        
        //Physics.CheckBox detects collisions between Box Colliders in unity, with the arguments 
        //(Center, size in each direction from center, rotation)
        while(Physics.CheckBox(newPos, size, newRot) || newPos.y < 0){
            //print($"COLLISION,Number of objects {objectsThisTime}, Framecount {frameCount}");
            //Generates new positon untill a collision is avoided
        
            screenPosition = maincam.ViewportToWorldPoint(new Vector3(Random.Range(0.2f,0.8f), Random.Range(0.2f,0.8f), Random.Range(20.0f,100.0f)));
            newPos = screenPosition;
        }
        


        var shape = pool.Get((ShapeLabel)prefabIndx);
        var newObj = shape.obj;

        newObj.transform.SetPositionAndRotation(newPos, newRot);

        
        
        
        //Saves object if it is a container
        //if (newObj.name.Contains("Container")){
        if (prefabIndx != 1){   

            MeshRenderer rend = newObj.GetComponentInChildren<MeshRenderer>();
            
            Vector3 center = new Vector3(4.5f,1.5f,-1.5f);
            Vector3 heading = newObj.transform.TransformPoint(center)-maincam.transform.position;
            float dist = heading.magnitude;
            Vector3 dir = heading/dist;
            
            if (Physics.Raycast(maincam.transform.position, dir, 100.0f)){
                //print($"HIT, Number of objects {objectsThisTime} Framecount {frameCount}");
                newObj.SetActive(false);
            }else{
                Dictionary<string,Vector3> screenCornerCord = FindCornersInScreen(newObj);
                saveGameObject(newObj, maincam.transform.position, rend, screenCornerCord);
            }
        }

        //Generates completely random color for objects generated 
        Color newColor = Random.ColorHSV();
        MeshRenderer gameObjectRenderer = newObj.GetComponent<MeshRenderer>();
        newObj.GetComponentInChildren<Renderer>().material.color = newColor;

        if (prefabIndx == 1){
            int rndInt = Random.Range(0,3);
            if (rndInt==2){
                newObj.SetActive(false);
            }
            
        }

        }
        synth.OnSceneChange();    
    }
   
   private Dictionary<string,Vector3> GetCornerCoordinates(GameObject gameObj){
        cornerCoordinates = new Dictionary<string,Vector3>();
        for (int i = 0; i < localCoord.Length; i++){
            cornerCoordinates[CornerLabel[i]] = gameObj.transform.TransformPoint(localCoord[i]);
        }
        return cornerCoordinates;
    }


    private class SaveObject{

        public string label;
        public string img_num;
        public Vector3 containerCameraPos;
        public Vector3 containerWorldPos;
        public float centerDistFromCamera;
        public Vector3 containerWorldRot;
        public Vector3 cameraWorldPos;
        public Vector3 cameraCameraPos;
        public  Vector3 containerWorldCenter;
        public Vector3 containerCameraCenter;
        public Vector3 screenCenterCoordinates;
        public Vector3 screenPositionCoordinates;
        public Vector3 cameraWorldRot;
        public string worldCornerCoordinates;

        public string screenCornerCoordinates;
    }


}

