import xml.etree.ElementTree as ET
import pyodbc
import os
from datetime import datetime
import logging
import traceback

# Set up logging with UTF-8 encoding to handle emojis
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xml_parser.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NamespaceAwareXMLParser:
    def __init__(self, server='SZLP112', database='Xml'):
        """Initialize the XML to SQL parser with namespace support"""
        self.server = server
        self.database = database
        self.connection = None
        self.cursor = None
        self.namespace = None  # Will be detected from XML
        self.stats = {
            'projects_processed': 0,
            'projects_inserted': 0,
            'baseline_projects_processed': 0,
            'baseline_projects_inserted': 0,
            'wbs_processed': 0,
            'wbs_inserted': 0,
            'activities_processed': 0,
            'activities_inserted': 0,
            'relationships_processed': 0,
            'relationships_inserted': 0,
            'errors': 0
        }
        
    def detect_namespace(self, root):
        """Detect XML namespace from root element"""
        if '}' in root.tag:
            # Extract namespace from root tag
            self.namespace = root.tag.split('}')[0] + '}'
            logger.info(f"[NS] Detected XML namespace: {self.namespace}")
        else:
            self.namespace = ''
            logger.info("[NS] No namespace detected")
        return self.namespace
    
    def ns_tag(self, tag_name):
        """Add namespace prefix to tag name if namespace exists"""
        if self.namespace:
            return f"{self.namespace}{tag_name}"
        return tag_name
    
    def connect_to_database(self):
        """Establish database connection"""
        try:
            connection_string = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={self.server};'
                f'DATABASE={self.database};'
                f'Trusted_Connection=yes;'
            )
            
            logger.info("[DB] Connecting to database...")
            self.connection = pyodbc.connect(connection_string)
            self.cursor = self.connection.cursor()
            
            # Test connection
            self.cursor.execute("SELECT @@VERSION, DB_NAME()")
            version_info = self.cursor.fetchone()
            logger.info("[DB] Connected to SQL Server successfully")
            logger.info(f"[DB] Database: {version_info[1]}")
            
            # Verify required tables exist
            self.cursor.execute("""
                SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'dbo'
                ORDER BY TABLE_NAME
            """)
            tables = [row[0] for row in self.cursor.fetchall()]
            logger.info(f"[DB] Available tables: {tables}")
            
            required_tables = ['Project', 'WBS', 'Activity', 'ActivityRelationships']
            missing_tables = [t for t in required_tables if t not in tables]
            
            if missing_tables:
                logger.error(f"[DB] Missing required tables: {missing_tables}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[DB] Database connection failed: {e}")
            return False
    
    def close_connection(self):
        """Close database connection and show statistics"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("[DB] Database connection closed")
        except Exception as e:
            logger.error(f"[DB] Error closing connection: {e}")
        
        # Show final statistics
        logger.info("=" * 60)
        logger.info("[STATS] FINAL PROCESSING STATISTICS")
        logger.info("=" * 60)
        for key, value in self.stats.items():
            logger.info(f"[STATS] {key.replace('_', ' ').title()}: {value}")
        logger.info("=" * 60)
    
    def safe_get_text(self, element, default=''):
        """Safely get text from XML element"""
        if element is not None and element.text is not None:
            result = element.text.strip()
            return result if result else default
        return default
    
    def safe_get_date(self, element):
        """Safely convert XML date to SQL datetime"""
        if element is not None and element.text is not None:
            try:
                date_text = element.text.strip()
                if date_text:
                    date_formats = [
                        '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S.%f',
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d',
                        '%m/%d/%Y',
                        '%d/%m/%Y'
                    ]
                    
                    for fmt in date_formats:
                        try:
                            return datetime.strptime(date_text, fmt)
                        except ValueError:
                            continue
                    
                    logger.warning(f"[DATE] Could not parse date: {date_text}")
            except Exception as e:
                logger.warning(f"[DATE] Date parsing error: {e}")
        return None
    
    def safe_get_float(self, element, default=0.0):
        """Safely convert XML text to float"""
        if element is not None and element.text is not None:
            try:
                text = element.text.strip()
                if text:
                    return float(text)
            except ValueError as e:
                logger.warning(f"[FLOAT] Could not convert to float: {element.text}")
        return default
    
    def safe_get_int(self, element, default=0):
        """Safely convert XML text to integer"""
        if element is not None and element.text is not None:
            try:
                text = element.text.strip()
                if text:
                    return int(float(text))
            except ValueError as e:
                logger.warning(f"[INT] Could not convert to int: {element.text}")
        return default
    
    def clear_tables(self):
        """Clear all tables"""
        try:
            tables = ['Project', 'WBS', 'Activity', 'ActivityRelationships']
            
            logger.info("[CLEAR] Current table row counts:")
            for table in tables:
                try:
                    self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = self.cursor.fetchone()[0]
                    logger.info(f"[CLEAR] {table}: {count:,} rows")
                except Exception as e:
                    logger.warning(f"[CLEAR] {table}: Could not get count - {e}")
            
            # Clear in reverse order (foreign keys)
            for table in reversed(tables):
                self.cursor.execute(f"DELETE FROM {table}")
                affected = self.cursor.rowcount
                logger.info(f"[CLEAR] Cleared {table}: {affected} rows deleted")
            
            self.connection.commit()
            logger.info("[CLEAR] All tables cleared successfully")
            
        except Exception as e:
            logger.error(f"[CLEAR] Error clearing tables: {e}")
            raise
    
    def debug_xml_structure(self, root):
        """Debug XML structure with namespace awareness"""
        logger.info("[DEBUG] Analyzing XML structure...")
        logger.info(f"[DEBUG] Root element: {root.tag}")
        
        # Show first few child elements
        children = list(root)
        logger.info(f"[DEBUG] Root has {len(children)} direct children")
        
        for i, child in enumerate(children[:5]):
            logger.info(f"[DEBUG] Child {i}: {child.tag}")
            grandchildren = list(child)
            if grandchildren:
                logger.info(f"[DEBUG]   Has {len(grandchildren)} grandchildren")
                for j, grandchild in enumerate(grandchildren[:3]):
                    logger.info(f"[DEBUG]     Grandchild {j}: {grandchild.tag}")
        
        # Look for project-related elements with namespace
        project_tag = self.ns_tag('Project')
        baseline_tag = self.ns_tag('BaselineProject')
        
        logger.info(f"[DEBUG] Looking for elements:")
        logger.info(f"[DEBUG]   Project tag: {project_tag}")
        logger.info(f"[DEBUG]   BaselineProject tag: {baseline_tag}")
        
        # Count elements
        all_elements = {}
        for elem in root.iter():
            # Remove namespace for counting
            clean_tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if clean_tag not in all_elements:
                all_elements[clean_tag] = 0
            all_elements[clean_tag] += 1
        
        logger.info("[DEBUG] Element counts (without namespace):")
        for tag, count in sorted(all_elements.items())[:20]:  # Show first 20
            logger.info(f"[DEBUG]   {tag}: {count}")
        
        # Specifically look for relationship elements
        relationship_tags = ['Relationship', 'ActivityRelationship', 'Relationships']
        logger.info("[DEBUG] Relationship element search:")
        for rel_tag in relationship_tags:
            count = all_elements.get(rel_tag, 0)
            logger.info(f"[DEBUG]   {rel_tag}: {count} found")
            
            if count > 0:
                # Show sample structure
                sample_elements = root.findall(f'.//{self.ns_tag(rel_tag)}')
                if sample_elements:
                    sample = sample_elements[0]
                    logger.info(f"[DEBUG]   Sample {rel_tag} structure:")
                    for child in list(sample)[:8]:  # Show first 8 fields
                        child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        value = child.text.strip() if child.text else 'None'
                        logger.info(f"[DEBUG]     {child_tag}: {value}")
        
        return all_elements
    
    def insert_project(self, project_element, project_type):
        """Insert project data"""
        try:
            if project_type == "Project":
                self.stats['projects_processed'] += 1
            else:
                self.stats['baseline_projects_processed'] += 1
            
            project_name = self.safe_get_text(project_element.find(self.ns_tag('Name')))
            project_id = self.safe_get_text(project_element.find(self.ns_tag('ObjectId')))
            baseline_project_id = self.safe_get_text(project_element.find(self.ns_tag('CurrentBaselineProjectObjectId')))
            
            logger.debug(f"[PROJECT] Name: '{project_name}', ID: '{project_id}', Type: '{project_type}'")
            
            if not project_id:
                logger.warning(f"[PROJECT] Skipping project without ObjectId: {project_name}")
                return None
            
            sql = """
            INSERT INTO Project (ProjectName, ProjectID, BaselineProjectId)
            VALUES (?, ?, ?)
            """
            
            self.cursor.execute(sql, (project_name, project_id, baseline_project_id))
            
            if project_type == "Project":
                self.stats['projects_inserted'] += 1
            else:
                self.stats['baseline_projects_inserted'] += 1
                
            logger.info(f"[PROJECT] Inserted {project_type}: '{project_name}' - ID: {project_id}")
            return project_id
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"[PROJECT] Error inserting project: {e}")
            return None
    
    def insert_wbs(self, wbs_element, project_type):
        """Insert WBS data"""
        try:
            self.stats['wbs_processed'] += 1
            
            wbs_name = self.safe_get_text(wbs_element.find(self.ns_tag('Name')))
            wbs_object_id = self.safe_get_text(wbs_element.find(self.ns_tag('ObjectId')))
            wbs_parent_id = self.safe_get_text(wbs_element.find(self.ns_tag('ParentObjectId')))
            sequence_number = self.safe_get_int(wbs_element.find(self.ns_tag('SequenceNumber')))
            project_id = self.safe_get_text(wbs_element.find(self.ns_tag('ProjectObjectId')))
            
            logger.debug(f"[WBS] Name: '{wbs_name}', ID: '{wbs_object_id}', ProjectType: '{project_type}'")
            
            if not wbs_object_id:
                logger.warning(f"[WBS] Skipping WBS without ObjectId: {wbs_name}")
                return None
            
            sql = """
            INSERT INTO WBS (WBSName, WBSObjectID, WBSParentId, SequenceNumber, ProjectId, ProjectType)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            self.cursor.execute(sql, (wbs_name, wbs_object_id, wbs_parent_id, sequence_number, project_id, project_type))
            self.stats['wbs_inserted'] += 1
            logger.info(f"[WBS] Inserted: '{wbs_name}' - ID: {wbs_object_id} - Type: {project_type}")
            return wbs_object_id
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"[WBS] Error inserting WBS: {e}")
            return None
    
    def insert_activity(self, activity_element, project_type):
        """Insert activity data"""
        try:
            self.stats['activities_processed'] += 1
            
            # Basic fields
            activity_name = self.safe_get_text(activity_element.find(self.ns_tag('Name')))
            activity_id = self.safe_get_text(activity_element.find(self.ns_tag('ObjectId')))
            wbs_id = self.safe_get_text(activity_element.find(self.ns_tag('WBSObjectId')))
            project_id = self.safe_get_text(activity_element.find(self.ns_tag('ProjectObjectId')))
            activity_type = self.safe_get_text(activity_element.find(self.ns_tag('Type')))
            calendar_id = self.safe_get_text(activity_element.find(self.ns_tag('CalendarObjectId')))
            
            # Dates
            planned_start_date = self.safe_get_date(activity_element.find(self.ns_tag('PlannedStartDate')))
            planned_finish_date = self.safe_get_date(activity_element.find(self.ns_tag('PlannedFinishDate')))
            start_date = self.safe_get_date(activity_element.find(self.ns_tag('StartDate')))
            finish_date = self.safe_get_date(activity_element.find(self.ns_tag('FinishDate')))
            actual_start_date = self.safe_get_date(activity_element.find(self.ns_tag('ActualStartDate')))
            actual_finish_date = self.safe_get_date(activity_element.find(self.ns_tag('ActualFinishDate')))
            remaining_early_start_date = self.safe_get_date(activity_element.find(self.ns_tag('RemainingEarlyStartDate')))
            remaining_early_finish_date = self.safe_get_date(activity_element.find(self.ns_tag('RemainingEarlyFinishDate')))
            remaining_late_start_date = self.safe_get_date(activity_element.find(self.ns_tag('RemainingLateStartDate')))
            remaining_late_finish_date = self.safe_get_date(activity_element.find(self.ns_tag('RemainingLateFinishDate')))
            
            # Numeric fields
            planned_duration = self.safe_get_float(activity_element.find(self.ns_tag('PlannedDuration')))
            actual_duration = self.safe_get_float(activity_element.find(self.ns_tag('ActualDuration')))
            percent_complete = self.safe_get_float(activity_element.find(self.ns_tag('PercentComplete')))
            remaining_duration = self.safe_get_float(activity_element.find(self.ns_tag('RemainingDuration')))
            
            # Additional text fields
            duration_type = self.safe_get_text(activity_element.find(self.ns_tag('DurationType')))
            p6_id = self.safe_get_text(activity_element.find(self.ns_tag('Id')))
            percent_complete_type = self.safe_get_text(activity_element.find(self.ns_tag('PercentCompleteType')))
            activity_status = self.safe_get_text(activity_element.find(self.ns_tag('Status')))
            
            logger.debug(f"[ACTIVITY] Name: '{activity_name}', ID: '{activity_id}', ProjectType: '{project_type}'")
            
            if not activity_id:
                logger.warning(f"[ACTIVITY] Skipping activity without ObjectId: {activity_name}")
                return None
            
            sql = """
            INSERT INTO Activity (
                ActivityName, ActivityId, WBSId, ProjectId, ActivityType, CalenderId,
                PlannedStartDate, PlannedFinishDate, PlannedDuration, StartDate, FinishDate,
                DurationType, ActualStartDate, ActualFinishDate, ActualDuration, P6Id,
                PercentCompleteType, PercentComplete, RemainingDuration, ActivityStatus,
                RemainingEarlyStartDate, RemainingEarlyFinishDate, RemainingLateStartDate,
                RemainingLateFinishDate, ProjectType
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.cursor.execute(sql, (
                activity_name, activity_id, wbs_id, project_id, activity_type, calendar_id,
                planned_start_date, planned_finish_date, planned_duration, start_date, finish_date,
                duration_type, actual_start_date, actual_finish_date, actual_duration, p6_id,
                percent_complete_type, percent_complete, remaining_duration, activity_status,
                remaining_early_start_date, remaining_early_finish_date, remaining_late_start_date,
                remaining_late_finish_date, project_type
            ))
            
            self.stats['activities_inserted'] += 1
            logger.info(f"[ACTIVITY] Inserted: '{activity_name}' - ID: {activity_id} - Type: {project_type}")
            return activity_id
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"[ACTIVITY] Error inserting activity: {e}")
            return None
    
    def insert_relationship(self, relationship_element, project_type):
        """Insert relationship data"""
        try:
            self.stats['relationships_processed'] += 1
            
            relationship_id = self.safe_get_text(relationship_element.find(self.ns_tag('ObjectId')))
            predecessor_project_id = self.safe_get_text(relationship_element.find(self.ns_tag('PredecessorProjectObjectId')))
            predecessor_activity_id = self.safe_get_text(relationship_element.find(self.ns_tag('PredecessorActivityObjectId')))
            successor_activity_id = self.safe_get_text(relationship_element.find(self.ns_tag('SuccessorActivityObjectId')))
            successor_project_id = self.safe_get_text(relationship_element.find(self.ns_tag('SuccessorProjectObjectId')))
            lag = self.safe_get_float(relationship_element.find(self.ns_tag('Lag')))
            relationship_type = self.safe_get_text(relationship_element.find(self.ns_tag('Type')))
            
            logger.debug(f"[REL] ID: '{relationship_id}', Type: '{relationship_type}', ProjectType: '{project_type}'")
            
            if not relationship_id:
                logger.warning("[REL] Skipping relationship without ObjectId")
                return None
            
            sql = """
            INSERT INTO ActivityRelationships (
                RelationshipId, PredecessorProjectId, PredecessorActivityId,
                SuccessorActivityId, SuccessorProjectId, Lag, RelationshipType, ProjectType
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.cursor.execute(sql, (
                relationship_id, predecessor_project_id, predecessor_activity_id,
                successor_activity_id, successor_project_id, lag, relationship_type, project_type
            ))
            
            self.stats['relationships_inserted'] += 1
            logger.info(f"[REL] Inserted: {relationship_id} ({relationship_type}) - Type: {project_type}")
            return relationship_id
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"[REL] Error inserting relationship: {e}")
            return None
    
    def process_project_container(self, container_element, project_type):
        """Process a Project or BaselineProject container"""
        logger.info(f"[PROCESS] Processing {project_type} container...")
        
        # Step 1: Process the Project itself
        project_id = self.insert_project(container_element, project_type)
        if not project_id:
            logger.warning(f"[PROCESS] Failed to insert {project_type}, skipping its children")
            return
        
        # Step 2: Process ALL WBS elements within this container
        wbs_elements = container_element.findall(f'.//{self.ns_tag("WBS")}')
        logger.info(f"[PROCESS] Found {len(wbs_elements)} WBS elements in {project_type}")
        
        for wbs in wbs_elements:
            self.insert_wbs(wbs, project_type)
        
        # Step 3: Process ALL Activity elements within this container
        activities = container_element.findall(f'.//{self.ns_tag("Activity")}')
        logger.info(f"[PROCESS] Found {len(activities)} Activity elements in {project_type}")
        
        for activity in activities:
            self.insert_activity(activity, project_type)
        
        # Step 4: Process ALL Relationship elements within this container
        # Look for both "Relationship" and "ActivityRelationship" elements
        relationships = container_element.findall(f'.//{self.ns_tag("Relationship")}')
        activity_relationships = container_element.findall(f'.//{self.ns_tag("ActivityRelationship")}')
        
        # Combine both types
        all_relationships = relationships + activity_relationships
        logger.info(f"[PROCESS] Found {len(relationships)} Relationship + {len(activity_relationships)} ActivityRelationship elements = {len(all_relationships)} total in {project_type}")
        
        for relationship in all_relationships:
            self.insert_relationship(relationship, project_type)
    
    def parse_xml_file(self, xml_file_path):
        """Main XML parsing method with namespace support"""
        try:
            logger.info("=" * 80)
            logger.info(f"[MAIN] STARTING XML PARSING: {xml_file_path}")
            logger.info("=" * 80)
            
            # Validate file
            if not os.path.exists(xml_file_path):
                raise FileNotFoundError(f"XML file not found: {xml_file_path}")
            
            file_size = os.path.getsize(xml_file_path)
            logger.info(f"[MAIN] File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
            if file_size == 0:
                raise ValueError("XML file is empty")
            
            # Parse XML
            logger.info("[MAIN] Parsing XML file...")
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            logger.info(f"[MAIN] XML parsed successfully. Root element: {root.tag}")
            
            # Detect and set namespace
            self.detect_namespace(root)
            
            # Debug XML structure
            self.debug_xml_structure(root)
            
            # Clear existing data
            logger.info("[MAIN] Clearing existing database data...")
            self.clear_tables()
            
            # Find Project containers with namespace
            project_tag = self.ns_tag('Project')
            baseline_tag = self.ns_tag('BaselineProject')
            
            project_containers = root.findall(f'.//{project_tag}')
            logger.info(f"[MAIN] Found {len(project_containers)} <{project_tag}> containers")
            
            baseline_containers = root.findall(f'.//{baseline_tag}')
            logger.info(f"[MAIN] Found {len(baseline_containers)} <{baseline_tag}> containers")
            
            if not project_containers and not baseline_containers:
                logger.error("[MAIN] No Project or BaselineProject containers found!")
                logger.info("[MAIN] Attempting to find elements without full namespace...")
                
                # Try alternative approaches
                for element in root.iter():
                    clean_tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
                    if clean_tag in ['Project', 'BaselineProject']:
                        logger.info(f"[MAIN] Found element: {element.tag}")
                        if clean_tag == 'Project':
                            project_containers.append(element)
                        else:
                            baseline_containers.append(element)
                
                if not project_containers and not baseline_containers:
                    logger.error("[MAIN] Still no project containers found after alternative search!")
                    return
            
            # Process each Project container
            for i, project_container in enumerate(project_containers, 1):
                logger.info(f"[MAIN] Processing Project container {i}/{len(project_containers)}")
                self.process_project_container(project_container, "Project")
            
            # Process each BaselineProject container
            for i, baseline_container in enumerate(baseline_containers, 1):
                logger.info(f"[MAIN] Processing BaselineProject container {i}/{len(baseline_containers)}")
                self.process_project_container(baseline_container, "BaselineProject")
            
            # Commit all changes
            logger.info("[MAIN] Committing all changes to database...")
            self.connection.commit()
            
            logger.info("=" * 80)
            logger.info("[MAIN] XML PARSING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
        except ET.ParseError as e:
            self.stats['errors'] += 1
            logger.error(f"[MAIN] XML parsing error: {e}")
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"[MAIN] Critical error during XML processing: {e}")
            logger.error(traceback.format_exc())
            
            if self.connection:
                logger.info("[MAIN] Rolling back database changes...")
                self.connection.rollback()

def view_database_stats(server='SZLP112', database='Xml'):
    """View current database statistics"""
    try:
        connection_string = (
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'Trusted_Connection=yes;'
        )
        
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()
        
        tables = ['Project', 'WBS', 'Activity', 'ActivityRelationships']
        
        print("\n[STATS] CURRENT DATABASE STATISTICS")
        print("="*50)
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table:20}: {count:,} rows")
            except Exception as e:
                print(f"{table:20}: Error - {e}")
        
        # Show ProjectType distribution
        print("\n[STATS] PROJECT TYPE DISTRIBUTION")
        print("="*50)
        
        for table in ['WBS', 'Activity', 'ActivityRelationships']:
            try:
                cursor.execute(f"""
                    SELECT ProjectType, COUNT(*) 
                    FROM {table} 
                    GROUP BY ProjectType 
                    ORDER BY ProjectType
                """)
                results = cursor.fetchall()
                print(f"\n{table}:")
                for project_type, count in results:
                    print(f"  {project_type}: {count:,}")
            except Exception as e:
                print(f"{table}: Error - {e}")
        
        print("="*50)
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"[STATS] Error connecting to database: {e}")

def main():
    """Main function"""
    xml_file_path = input("[INPUT] Enter the path to your XML file: ").strip().strip('"\'')
    
    # Validate file
    if not os.path.exists(xml_file_path):
        print(f"[ERROR] File not found: {xml_file_path}")
        return
    
    file_size = os.path.getsize(xml_file_path)
    if file_size == 0:
        print("[ERROR] File is empty!")
        return
    
    print(f"[INFO] File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Initialize parser
    parser = NamespaceAwareXMLParser(server='SZLP112', database='Xml')
    
    try:
        # Connect to database
        if not parser.connect_to_database():
            print("[ERROR] Failed to connect to database.")
            return
        
        # Parse XML file
        parser.parse_xml_file(xml_file_path)
        
        print("\n[SUCCESS] XML parsing completed!")
        print("[INFO] Check 'xml_parser.log' for detailed processing information.")
        
        # Show final statistics
        view_database_stats()
        
    except Exception as e:
        print(f"[ERROR] {e}")
        logger.exception("Fatal error in main")
        
    finally:
        parser.close_connection()

if __name__ == "__main__":
    main()